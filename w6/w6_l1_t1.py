import numpy
import math
from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Загрузите картинку parrots.jpg.
image = imread('data/parrots.jpg')
#print(image)

# Преобразуйте изображение, приведя все значения в интервал от 0 до 1. Для этого можно воспользоваться функцией img_as_float из модуля skimage.
image_float = img_as_float(image)
#print(image_float)

# 2. Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя координатами - значениями интенсивности в пространстве RGB.
r = image_float[:, :, 0].ravel()
g = image_float[:, :, 1].ravel()
b = image_float[:, :, 2].ravel()
rgb = numpy.transpose(numpy.vstack((r, g, b)))

# 3. Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241.
kM = KMeans(init='k-means++', random_state=241)
kM.fit(rgb)
cl = kM.labels_

# После выделения кластеров все пиксели, отнесенные в один кластер, попробуйте заполнить двумя способами:
# медианным и средним цветом по кластеру.
colors_avg = kM.cluster_centers_
print(colors_avg)

cl_img = numpy.reshape(cl, (-1, 713))
img_new = numpy.copy(image_float)
for cluster in range(0, kM.n_clusters):
    #mean_r = numpy.median(img_new[:, :, 0][cl_img == cluster])
    #mean_g = numpy.median(img_new[:, :, 1][cl_img == cluster])
    #mean_b = numpy.median(img_new[:, :, 2][cl_img == cluster])
    # print [mean_r, mean_g, mean_b]

    img_new[cl_img == cluster] = colors_avg[cluster]
    plt.imshow(img_new)

img_new_1 = numpy.copy(image_float)
for cluster in range(0, kM.n_clusters):
    median_r = numpy.median(img_new[:, :, 0][cl_img == cluster])
    median_g = numpy.median(img_new[:, :, 1][cl_img == cluster])
    median_b = numpy.median(img_new[:, :, 2][cl_img == cluster])
    print([median_r, median_g, median_b])

    img_new_1[cl_img == cluster] = [median_r, median_g, median_b]
    plt.imshow(img_new_1)


# 3. Измерьте качество получившейся сегментации с помощью метрики PSNR. Эту метрику нужно реализовать самостоятельно
def PSNR(image1, image2):
    """Function calculates PSNR metrics between two images"""
    mse = numpy.mean((image1 - image2) ** 2)
    psnr = 10 * math.log10(numpy.max(image1) / mse)
    return psnr

print(PSNR(image_float, img_new))

# 4. Найдите минимальное количество кластеров, при котором значение PSNR выше 20 (можно рассмотреть не более
# 20 кластеров). Это число и будет ответом в данной задаче.
for i in range(1, 21):
    kM = KMeans(n_clusters = i, init = 'k-means++', random_state = 241)
    kM.fit(rgb)
    cl = kM.labels_
    colors_avg = kM.cluster_centers_
    cl_img = numpy.reshape(cl, (-1, 713))
    img_new = numpy.copy(image_float)
    for cluster in range(0, i):
        img_new[cl_img == cluster] = colors_avg[cluster]
    print (i, PSNR(image_float, img_new))
    plt.figure()
    plt.imshow(img_new)

