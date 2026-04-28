import copy
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from src.evaluation import evaluate

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, n_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        lstm_out_size = hidden_size * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_size, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(lstm_out_size)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm(lstm_out + attn_out)
        out = self.fc(attn_out[:, -1, :])
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=2, dropout=0.2, dim_feedforward=256):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])
        return out


def build_grouped_sequences(df, feature_cols, target_col, seq_length, group_cols=None):
    if group_cols is None:
        group_cols = ["ref_branch_code", "material_nature_sum_desc"]
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        raise ValueError("分组列不存在")
    X_seq, y_seq = [], []
    for _, group_df in df.groupby(group_cols, sort=False):
        group_df = group_df.sort_values("month").reset_index(drop=True)
        n = len(group_df)
        if n <= seq_length:
            continue
        missing = [c for c in feature_cols if c not in group_df.columns]
        if missing:
            continue
        X_g = group_df[feature_cols].values.astype(np.float32)
        y_g = group_df[target_col].values.astype(np.float32)
        if np.isnan(X_g).any() or np.isnan(y_g).any():
            continue
        for i in range(n - seq_length):
            X_seq.append(X_g[i : i + seq_length])
            y_seq.append(y_g[i + seq_length])
    if len(X_seq) == 0:
        raise ValueError(f"未能构造任何序列！seq_length={seq_length}")
    return np.array(X_seq), np.array(y_seq)


def prepare_dl_features(train_raw, test_raw):
    for df in [train_raw, test_raw]:
        df["month"] = pd.to_datetime(df["month"])
        if "year" not in df.columns:
            df["year"] = df["month"].dt.year
        if "month_num" not in df.columns:
            df["month_num"] = df["month"].dt.month
        if "quarter" not in df.columns:
            df["quarter"] = df["month"].dt.quarter
        if "month_sin" not in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        if "month_cos" not in df.columns:
            df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    cat_feats = [
        "Business_Type_1_Desc", "Type_Group_Desc", "Shop_Style_Desc",
        "City_Description", "City", "Province_Description",
        "City_Level_Description", "District_Desc", "Geographic_Region_Desc",
        "Mall_Scale_Code_Desc", "shop_type_desc",
    ]
    cat_feats = [c for c in cat_feats if c in train_raw.columns]
    holiday_feats = [c for c in train_raw.columns if c.startswith("holiday_") or c == "total_holiday_days"]
    time_feats = ["year", "month_num", "quarter", "month_sin", "month_cos"]
    time_feats = [c for c in time_feats if c in train_raw.columns]

    for col in cat_feats:
        le = LabelEncoder()
        combined = pd.concat([train_raw[col], test_raw[col]], ignore_index=True).astype(str)
        le.fit(combined)
        train_raw[col] = le.transform(train_raw[col].astype(str))
        test_raw[col] = le.transform(test_raw[col].astype(str))

    price_feat = ["price"] if "price" in train_raw.columns else []
    feature_cols = time_feats + holiday_feats + cat_feats + price_feat
    for c in ["store_code", "prod_code"]:
        if c in train_raw.columns and c not in feature_cols:
            feature_cols.append(c)
    return feature_cols


def train_dl_model(model, train_loader, val_loader, device, epochs=100, patience=15, lr=0.001, weight_decay=1e-5):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 3),
    )
    best_val_loss = float("inf")
    best_model_weights = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"DL early stop at epoch {epoch+1}, best val loss {best_val_loss:.6f}")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    return model


def run_dl_experiment(
    train_raw, test_raw, model_type="lstm",
    seq_length=12, epochs=100, batch_size=256, patience=15,
    lr=0.001, weight_decay=1e-5, hidden_size=128, d_model=128,
) -> Tuple[np.ndarray, float, float, nn.Module]:
    logger.info(f"Training {model_type.upper()} (seq_length={seq_length})")

    feature_cols = prepare_dl_features(train_raw, test_raw)
    X_train, y_train = build_grouped_sequences(train_raw, feature_cols, "monthly_sales", seq_length)
    X_test, y_test = build_grouped_sequences(test_raw, feature_cols, "monthly_sales", seq_length)

    logger.info(f"Samples: Train={X_train.shape[0]:,}, Test={X_test.shape[0]:,}")

    scaler_X = StandardScaler()
    n_train, seq_len, n_feats = X_train.shape
    X_train_flat = scaler_X.fit_transform(X_train.reshape(-1, n_feats))
    X_train = X_train_flat.reshape(n_train, seq_len, n_feats)
    X_test_flat = scaler_X.transform(X_test.reshape(-1, n_feats))
    X_test = X_test_flat.reshape(-1, seq_len, n_feats)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    val_size = int(len(X_train) * 0.15)
    if val_size < 10:
        val_size = min(100, len(X_train) // 5)
    X_val, y_val = X_train[-val_size:], y_train_scaled[-val_size:]
    X_train_split, y_train_split = X_train[:-val_size], y_train_scaled[:-val_size]

    train_dataset = TensorDataset(torch.FloatTensor(X_train_split), torch.FloatTensor(y_train_split))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if model_type.lower() == "lstm":
        model = LSTMAttentionModel(input_size=n_feats, hidden_size=hidden_size, num_layers=2, dropout=0.2, n_heads=4)
    elif model_type.lower() == "transformer":
        model = TransformerModel(input_size=n_feats, d_model=d_model, nhead=4, num_layers=2, dropout=0.2, dim_feedforward=d_model * 2)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    model = train_dl_model(model, train_loader, val_loader, device, epochs=epochs, patience=patience, lr=lr, weight_decay=weight_decay)

    model.eval()
    all_preds = []
    test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i + batch_size]
            preds = model(batch).cpu().numpy()
            all_preds.append(preds)
    y_pred_scaled = np.concatenate(all_preds).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_pred = np.maximum(0, y_pred)

    assert len(y_test) == len(y_pred), f"Length mismatch: y_test={len(y_test)}, y_pred={len(y_pred)}"
    metrics = evaluate(y_test, y_pred)
    logger.info(f"{model_type.upper()} Test MAPE={metrics['mape']:.2f}% WAPE={metrics['wape']:.2f}%")
    return y_pred, metrics["mape"], metrics["wape"], model
