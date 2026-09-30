# UVO SLAM: Керівництво користувача для NVIDIA Jetson

Компактне та неінвазивне розгортання UVO SLAM на бортових комп'ютерах **NVIDIA Jetson (Orin Nano / NX / AGX Orin / Xavier)**.  
Скрипти працюють **виключно в user-space**: без модифікації файлів підкачки ОС, без `sudo`, без примусових перевстановлень PyTorch та без створення системних демонів.

---

## 🚀 1. Швидке встановлення залежностей та офлайн-кешування (1 команда)

У терміналі Jetson виконайте:

```bash
chmod +x deploy_jetson.sh
./deploy_jetson.sh
```

### Що робить скрипт:
1. **Перевірка оточення:** Переконується у доступності CUDA на вашому Jetson.
2. **Встановлення Python-бібліотек:** Доставляє необхідні пакети (`timm`, `einops`, `scipy`, `huggingface_hub`).
3. **Офлайн-кешування нейромереж (Zero-Internet):** Завантажує моделі XFeat, DINOv2 та UniDepth V2 у локальний кеш. У полі ровер зможе стартувати **взагалі без інтернету чи зв'язку**.

---

## 📱 2. Тактичний Веб-Дашборд на планшет оператора (`--web`)

Головна фіча для польової експлуатації: ровер запускає потік, а оператор на планшеті, смартфоні чи ноутбуці відкриває браузер за адресою:
```text
http://<IP_АДРЕСА_JETSON>:8080
```

### Що відображається на планшеті:
* **Живе відео камери (30 FPS)** із зеленими векторами відстеження ключових точок.
* **Врізка метричної глибини (UniDepth V2)** у псевдокольорах.
* **Жива 2.5D карта прохідності (BEV Costmap)**:
  * Зелена зона — безпечна колія дороги;
  * Жовта — купини;
  * Червона — каміння, стовбури, стіни;
  * Блакитна — вирви, ями та урвища;
  * Розмітка кожні 2 метри та жовтий контур корпусу ровера.
* **Телеметрія реального часу:** локальні координати $(X, Y, Z)$, інлаєри, статус одометрії, FPS.
* **Інтерактивний блок ⚓ Geo-Anchor:**
  * Оператор може прямо з планшета ввести координати орієнтира $(\text{Lat}, \text{Lon}, \text{Heading})$ і натиснути **"Lock Anchor"**.
  * Кнопка **"📍 My GPS"**: використовує вбудований GPS планшета оператора в 1 клік!

---

## 📹 3. Команди запуску з різними типами камер

### Варіант А: Звичайна USB-вебкамера (`/dev/video0`)
```bash
python3 UVO/scripts/run_uvo_live_camera.py --type usb --source 0 --web
```

### Варіант Б: CSI-камера на шлейфі (IMX219 / IMX477)
Використовує апаратний акселератор GStreamer `nvarguscamerasrc`:
```bash
python3 UVO/scripts/run_uvo_live_camera.py --type csi --source 0 --width 1280 --height 720 --fps 30 --web
```

### Варіант В: Мережева IP-камера (Dahua / Hikvision по RTSP)
Використовує апаратне декодування Jetson NVDEC (`nvv4l2decoder`):
```bash
python3 UVO/scripts/run_uvo_live_camera.py --type rtsp --source "rtsp://admin:pass@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0" --web
```

### Варіант Г: Тестовий запуск на збереженому відеофайлі
```bash
python3 UVO/scripts/run_uvo_live_camera.py --video front_190610.mp4 --web
```

---

## ⚡ 4. Повний стек оптимізацій продуктивності (Threaded SLAM)

1. **Двопоточна архітектура (Decoupled Threaded SLAM):**
   * **Потік 1 (Tracking Thread):** Обробляє камери на частоті **60–80 FPS** (XFeat + Fast PnP). Жодного блокування чи очікування нейромереж!
   * **Потік 2 (Async Mapping Worker):** У фоні асинхронно рахує метричну глибину UniDepth V2, оптимізує граф поз BA та будує 2.5D Costmap.
2. **Асинхронний Web-кодувальник:**
   * Рендеринг тактичного дашборду та стиснення в JPEG перенесено у фоновий потік `AsyncDashboardEncoder`. Основний цикл трекінгу витрачає 0 мс на веб-сервер.
3. **FP16 Half-Precision (Tensor Cores):**
   * Увімкнено за замовчуванням (`--fp16`). 2x прискорення нейромереж та 50% економія VRAM.
4. **Масштабування роздільної здатності UniDepth (`max_resolution=518`):**
   * Зменшує кількість оброблюваних пікселів у 6 разів без втрати метричної точності масштабу.
5. **Субдискретизація карти прохідності (`subsample_step=3`):**
   * Розрахунок 2.5D сітки займає < 3 мс.

---

## 🚀 5. Експорт у нативні рушії TensorRT (`.engine`)

Для отримання максимального апаратного прискорення на JetPack:
```bash
python3 UVO/scripts/export_tensorrt.py --model all --output_dir tensorrt_models
```
Після чого скомпілюйте бінарний рушій під ваш чіп Orin/Xavier:
```bash
/usr/src/tensorrt/bin/trtexec --onnx=tensorrt_models/dinov2_vits14.onnx --saveEngine=tensorrt_models/dinov2_vits14_fp16.engine --fp16
/usr/src/tensorrt/bin/trtexec --onnx=tensorrt_models/xfeat.onnx --saveEngine=tensorrt_models/xfeat_fp16.engine --fp16
```
