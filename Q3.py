import cv2
import numpy as np

# =========================
# 1. LEITURA
# =========================
img = cv2.imread("gatos.png")
if img is None:
    raise ValueError("Imagem não encontrada!")

original = img.copy()

# =========================
# 2. PRÉ-PROCESSAMENTO
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, binary = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

cv2.imwrite("01_binary.png", binary)

# =========================
# 3. MORFOLOGIA
# =========================
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imwrite("02_opening.png", opening)

sure_bg = cv2.dilate(opening, kernel, iterations=3)
cv2.imwrite("03_sure_bg.png", sure_bg)

# =========================
# 4. DISTANCE TRANSFORM
# =========================
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
dist_norm = np.uint8(dist_norm)

cv2.imwrite("04_distance.png", dist_norm)

_, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

cv2.imwrite("05_sure_fg.png", sure_fg)

# =========================
# 5. UNKNOWN REGION
# =========================
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imwrite("06_unknown.png", unknown)

# =========================
# 6. MARCADORES
# =========================
num_labels, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1
markers[unknown == 255] = 0

# =========================
# 7. WATERSHED
# =========================
markers_ws = markers.copy()
markers_ws = cv2.watershed(original, markers_ws)

# =========================
# 8. RESULTADO FINAL
# =========================
result = original.copy()
result[markers_ws == -1] = [0, 0, 255]

cv2.imwrite("07_watershed_result.png", result)

# =========================
# 9. SEGMENTAÇÃO COLORIDA
# =========================
segmented = np.zeros_like(original)

for label in np.unique(markers_ws):
    if label <= 1:
        continue
    segmented[markers_ws == label] = np.random.randint(0, 255, 3)

cv2.imwrite("08_segmented_regions.png", segmented)

# =========================
# 10. (OPCIONAL) EXIBIÇÃO
# =========================
cv2.imshow("Result", result)
cv2.imshow("Segmented", segmented)

cv2.waitKey(0)
cv2.destroyAllWindows()