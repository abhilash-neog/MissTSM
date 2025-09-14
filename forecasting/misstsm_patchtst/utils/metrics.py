import numpy as np


def RSE(pred, true, mask):
    pred = pred[mask]
    true = true[mask]
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true, mask):
    print(f"mask shape = {mask.shape}")
    print(f"pred shape = {pred.shape}")
    pred = pred[mask]
    true = true[mask]
    return np.mean(np.abs(pred - true))


def MSE(pred, true, mask):
    pred = pred[mask]
    true = true[mask]
    return np.mean((pred - true) ** 2)


def RMSE(pred, true, mask):
    return np.sqrt(MSE(pred, true, mask))


def MAPE(pred, true, mask):
    pred = pred[mask]
    true = true[mask]
    return np.mean(np.square((pred - true) / true))


def MSPE(pred, true, mask):
    pred = pred[mask]
    true = true[mask]
    return np.mean(np.square((pred - true) / true))


def metric(pred, true, mask):
    mask = mask.astype(bool)
    mae = MAE(pred, true, mask)
    mse = MSE(pred, true, mask)
    rmse = RMSE(pred, true, mask)
    mape = MAPE(pred, true, mask)
    mspe = MSPE(pred, true, mask)
    rse = RSE(pred, true, mask)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr
