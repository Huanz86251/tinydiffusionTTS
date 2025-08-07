import numpy as np
import matplotlib.pyplot as plt

# 加载npy文件
z_noise = np.load('./stft_test_output/z_noise.npy')
y_original = np.load('./stft_test_output/y_original.npy')
y_hat_denoise = np.load('./stft_test_output/y_hat_denoise.npy')

# 可视化mel谱图
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(z_noise, aspect='auto', origin='lower', vmin=-3, vmax=3)  # 固定到噪声范围
plt.title('Z Noise')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(y_original, aspect='auto', origin='lower')
plt.title('Y Original')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(y_hat_denoise, aspect='auto', origin='lower')
plt.title('Y Hat Denoise')
plt.colorbar()

plt.tight_layout()
plt.savefig("output_comparison.png", dpi=1000)
plt.show()
corr = np.corrcoef(z_noise.flatten(), y_original.flatten())[0,1]
print(f"[Correlation] z_noise and y_original: {corr:.4f}")

