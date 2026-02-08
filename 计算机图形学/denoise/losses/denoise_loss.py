import jittor as jt


def denoise_l1_loss(pred: jt.Var, target: jt.Var) -> jt.Var:
    """简单的 L1 损失，稳定且收敛快。"""
    return jt.mean(jt.abs(pred - target))


def calc_psnr(pred: jt.Var, target: jt.Var) -> float:
    """
    计算 PSNR，输入需已在 [0,1]。
    返回 Python float，便于日志打印。
    """
    mse = jt.mean((pred - target) ** 2)
    mse = jt.maximum(mse, 1e-8)
    psnr = 10.0 * jt.log(1.0 / mse) / jt.log(jt.float32(10.0))
    return float(psnr.item())

