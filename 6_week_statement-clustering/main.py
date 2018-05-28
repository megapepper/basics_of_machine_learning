import pandas as pd
import numpy as np
import skimage
from skimage.io import imread, imsave
from sklearn.cluster import KMeans

# 1. Загрузите картинку parrots.jpg. Преобразуйте изображение, приведя все значения в интервал от 0 до 1.
# Обратите внимание на этот шаг, так как при работе с исходным изображением вы получите некорректный результат.
image = skimage.img_as_float(imread('parrots.jpg'))

# 2. Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя координатами - значениями интенсивности
#  в пространстве RGB.
sh1, sh2, sh3 = image.shape
DF_pixels = pd.DataFrame(np.reshape(image, (sh1*sh2, sh3)), columns=['Red', 'Green', 'Blue'])

# 3. Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241. После выделения кластеров все
# пиксели, отнесенные в один кластер, попробуйте заполнить двумя способами: медианным и средним цветом по кластеру.
def cluster(pixels, n_clusters):
    pixels = pixels.copy()
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    pixels['cluster'] = model.fit_predict(pixels)
    # средний цвет по кластеру
    means = pixels.groupby('cluster').mean().values
    mean_pixels = [means[c] for c in pixels['cluster'].values]
    mean_image = np.reshape(mean_pixels, (sh1, sh2, sh3))
    imsave('images/parrots_mean'+str(n_clusters)+'.jpg',mean_image)
    # медианный цвет по кластеру
    medians = pixels.groupby('cluster').median().values
    median_pixels = [medians[c] for c in pixels['cluster'].values]
    median_image = np.reshape(median_pixels, (sh1, sh2, sh3))
    imsave('images/parrots_median'+str(n_clusters)+'.jpg',median_image)
    return mean_image, median_image

# 4. Измерьте качество получившейся сегментации с помощью метрики PSNR. Эту метрику нужно реализовать
# самостоятельно (см. определение).
def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return 10 * np.math.log10(float(1) / mse)

# 5. Найдите минимальное количество кластеров, при котором значение PSNR выше 20 (можно рассмотреть
# не более 20кластеров). Это число и будет ответом в данной задаче.
for n_clusters in range(1, 21):
    mean_image, median_image = cluster(DF_pixels, n_clusters)
    psnr_mean, psnr_median = psnr(image, mean_image), psnr(image, median_image)
    #print (psnr_mean, psnr_median)

    if psnr_mean > 20 or psnr_median > 20:
        print('n_clusters=', n_clusters)
        break

