import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error, root_mean_squared_error


class HeatmapComparison:
    def __init__(self, heatmap1: np.ndarray, heatmap2: np.ndarray):
        self.heatmap1 = heatmap1
        self.heatmap2 = heatmap2

    def MSE(self) -> float:
        # Flatten since mean_squared_error only works with 1D data
        mse = mean_squared_error(self.heatmap1.flatten(), self.heatmap2.flatten())
        return mse

    def RMSE(self) -> float:
        # Flatten since root_mean_squared_error only works with 1D data
        rmse = root_mean_squared_error(self.heatmap1.flatten(), self.heatmap2.flatten())
        return rmse

    def SSIM(self) -> float:
        ssim = structural_similarity(self.heatmap1, self.heatmap2, data_range=1.0)
        return ssim

    def PSNR(self) -> float:
        mse = self.MSE()
        psnr = 10 * np.log10(1 / mse)
        return psnr
