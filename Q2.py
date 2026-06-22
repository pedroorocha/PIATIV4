import cv2
import numpy as np

# =========================
# 1. LEITURA
# =========================
img = cv2.imread("gato.jpg")
if img is None:
    raise ValueError("Imagem não encontrada!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# 2. SUAVIZAÇÃO GAUSSIANA
# =========================
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
cv2.imwrite("01_gaussian.png", blur)

# =========================
# 3. SOBEL MANUAL (GRADIENTE)
# =========================

# kernels Sobel
sobel_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])

h, w = blur.shape

gx = np.zeros_like(blur, dtype=np.float32)
gy = np.zeros_like(blur, dtype=np.float32)

# convolução manual
for i in range(1, h - 1):
    for j in range(1, w - 1):
        region = blur[i-1:i+2, j-1:j+2]

        gx[i, j] = np.sum(region * sobel_x)
        gy[i, j] = np.sum(region * sobel_y)

# =========================
# 4. MAGNITUDE E ORIENTAÇÃO
# =========================
magnitude = np.sqrt(gx**2 + gy**2)
angle = np.arctan2(gy, gx) * (180 / np.pi)
angle[angle < 0] += 180  # HOG usa 0–180

# normalizar magnitude para salvar imagem
mag_img = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
mag_img = np.uint8(mag_img)

cv2.imwrite("02_magnitude.png", mag_img)

# =========================
# 5. HOG SIMPLIFICADO
# =========================

cell_size = 8
bins = 9
bin_range = 180 / bins

h_cells = h // cell_size
w_cells = w // cell_size

hog = np.zeros((h_cells, w_cells, bins))

for i in range(h_cells):
    for j in range(w_cells):

        cell_mag = magnitude[
            i*cell_size:(i+1)*cell_size,
            j*cell_size:(j+1)*cell_size
        ]

        cell_ang = angle[
            i*cell_size:(i+1)*cell_size,
            j*cell_size:(j+1)*cell_size
        ]

        hist = np.zeros(bins)

        for x in range(cell_size):
            for y in range(cell_size):
                m = cell_mag[x, y]
                a = cell_ang[x, y]

                bin_idx = int(a // bin_range)
                bin_idx = min(bin_idx, bins - 1)

                hist[bin_idx] += m

        hog[i, j] = hist

# =========================
# 6. VISUALIZAÇÃO SIMPLES DO HOG
# =========================

hog_vis = np.zeros_like(gray)

for i in range(h_cells):
    for j in range(w_cells):
        strength = np.sum(hog[i, j])

        cv2.rectangle(
            hog_vis,
            (j*cell_size, i*cell_size),
            ((j+1)*cell_size, (i+1)*cell_size),
            int(strength / 10),
            -1
        )

cv2.imwrite("03_hog.png", hog_vis)

# =========================
# 7. EXIBIÇÃO
# =========================
cv2.imshow("Gaussian", blur)
cv2.imshow("Magnitude", mag_img)
cv2.imshow("HOG", hog_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()