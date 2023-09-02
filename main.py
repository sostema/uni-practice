import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import global_max_pool as gmp, global_add_pool as gap


# Загрузка данных
def load_data():
    click_df = pd.read_pickle("./data/yoochoose-clicks.dat")
    buy_df = pd.read_pickle("./data/yoochoose-buys.dat")
    buy_df.columns = ["session_id", "timestamp", "item_id", "price", "quantity"]

    click_df["valid_session"] = click_df["session_id"].map(
        click_df.groupby("session_id")["item_id"].transform("count") > 2
    )

    click_df = click_df[click_df["valid_session"]].drop("valid_session", axis=1)

    sampled_sessions = np.random.choice(
        click_df["session_id"].unique(), 1000000, replace=False
    )

    click_df = click_df[click_df["session_id"].isin(sampled_sessions)]

    item_encoder = LabelEncoder()
    click_df["item_id"] = item_encoder.fit_transform(click_df["item_id"])

    click_df["label"] = click_df["session_id"].isin(buy_df["session_id"])

    return click_df, item_encoder


# Создание PyTorch Geometric Dataset
def create_dataset(df, item_encoder):
    data_list = []

    grouped = df.groupby("session_id")
    for session_id, group in tqdm(grouped):
        sess_item_id = LabelEncoder().fit_transform(group.item_id)
        group = group.reset_index(drop=True)
        group["sess_item_id"] = sess_item_id
        node_features = (
            group.loc[group.session_id == session_id, ["sess_item_id", "item_id"]]
            .sort_values("sess_item_id")
            .item_id.drop_duplicates()
            .values
        )

        node_features = torch.LongTensor(node_features).unsqueeze(1)
        target_nodes = group.sess_item_id.values[1:]
        source_nodes = group.sess_item_id.values[:-1]

        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        x = node_features

        y = torch.FloatTensor([group.label.values[0]])

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list


# Определение PyTorch Geometric модели
class RecommenderModel(torch.nn.Module):
    def __init__(self, num_items, embedding_dim=128):
        super(RecommenderModel, self).__init__()
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, 128)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.lin1(x1)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x


# Собственный SAGEConv слой
class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr="max")  #  "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(
            in_channels + out_channels, in_channels, bias=False
        )
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        return x_j

    def update(self, aggr_out, x):
        new_embedding = torch.cat([aggr_out, x], dim=1)
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        return new_embedding


# Тренировка модели
def train_model(model, train_loader, val_loader, num_epochs=1, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            labels = data.y.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_auc = evaluate_auc(model, train_loader)
        val_auc = evaluate_auc(model, val_loader)

        print(
            f"""Epoch [{epoch + 1}/{num_epochs}]
            - Loss: {avg_loss:.4f}
            - Train AUC: {train_auc:.4f}
            - Val AUC: {val_auc:.4f}"""
        )


# Оценка модели по метрике AUC
def evaluate_auc(model, data_loader):
    device = "mps"
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            outputs = model(data).detach().cpu().numpy()
            labels = data.y.detach().cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(outputs)

    return roc_auc_score(y_true, y_pred)


# Основная функция
def main():
    click_df, item_encoder = load_data()
    data_list = create_dataset(click_df, item_encoder)

    train_size = int(0.8 * len(data_list))
    val_size = int(0.1 * len(data_list))
    (
        train_data,
        val_data,
    ) = (
        data_list[:train_size],
        data_list[train_size : train_size + val_size],
    )

    train_loader = GeoDataLoader(train_data, batch_size=1024, shuffle=True)
    val_loader = GeoDataLoader(val_data, batch_size=1024)

    num_items = click_df["item_id"].nunique()
    model = RecommenderModel(num_items)
    train_model(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
