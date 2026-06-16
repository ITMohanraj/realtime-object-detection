# 👁️ Vision Assistant Deployment Guide

This guide provides step-by-step instructions to run, build, and deploy the Blind People Object Detection project as a live web application.

---

## 🛠️ Local Installation & Development

### 1. Standalone Python Setup

To run the application locally without Docker, follow these steps:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you plan to run the original desktop app (`main.py`), you must also install Streamlit and Pygame:*
   ```bash
   pip install streamlit pygame
   ```

2. **Download Weights (Optional):**
   The application will automatically download files when it starts. However, you can run the helper downloader manually:
   ```bash
   python download_weights.py
   ```

3. **Start the FastAPI Server:**
   ```bash
   python app.py
   ```
   Or run using Uvicorn directly:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

4. **Access the Web App:**
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser. You will see a beautiful dark-mode interface with live webcam support, canvas drawings, and sound cues.
   
5. **Interactive API Docs:**
   Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to view the automated Swagger/OpenAPI documentation.

---

## 🐳 Containerized Execution (Docker)

To run the application in a local container (useful to match production behavior exactly):

### Option A: Using Raw Docker Commands
1. **Build the Image:**
   ```bash
   docker build -t realtime-object-detection .
   ```
2. **Run the Container:**
   ```bash
   docker run -d -p 10000:10000 --name realtime-object-detection-server realtime-object-detection
   ```
3. **Open Browser:**
   Go to [http://localhost:10000](http://localhost:10000).

### Option B: Using Docker Compose
1. **Launch Stack:**
   ```bash
   docker-compose up -d --build
   ```
2. **Stop Stack:**
   ```bash
   docker-compose down
   ```

---

## 🚀 Live Cloud Deployment on Render (Free Tier)

You can host this backend for free on [Render](https://render.com).

### Option A: One-Click Blueprint Deployment (Recommended)
1. Commit all project files (including `Dockerfile`, `render.yaml`, `requirements.txt`, etc.) to your GitHub repository.
2. Log in to the [Render Dashboard](https://dashboard.render.com/).
3. Click **Blueprints** -> **New Blueprint Instance**.
4. Connect your GitHub repository.
5. Render will automatically detect the `render.yaml` file, spin up the Docker builder, and provision your application on the free tier.

### Option B: Manual Web Service Deployment
1. Log in to the [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** -> **Web Service**.
3. Link your GitHub repository.
4. Set the following options:
   * **Language:** `Docker`
   * **Branch:** `main` (or your default branch)
   * **Instance Type:** `Free` (512MB RAM, CPU)
5. Open **Advanced** configurations and add the following Environment Variables:
   * `MODEL_TYPE` = `yolov3-tiny` (Ensures the server stays within the 512MB RAM limit)
   * `DOWNLOAD_FULL_YOLO` = `false`
   * `CONFIDENCE_THRESHOLD` = `0.5`
6. Click **Deploy Web Service**.

*Note: The Render free tier automatically puts services to sleep after 15 minutes of inactivity. The first request after a sleep period may take 1-2 minutes to spin up.*

---

## 🔌 API Integration & Android Connection

If you are connecting an **Android app** or a separate frontend client, use the following endpoint specification:

### 1. Health Check
* **Endpoint:** `GET /api/health`
* **Response:**
  ```json
  {
    "status": "healthy",
    "timestamp": 1700000000.0,
    "configuration": {
      "model_type": "yolov3-tiny",
      "cfg_file": "yolov3-tiny.cfg",
      "weights_file": "yolov3-tiny.weights"
    },
    "models_loaded": true
  }
  ```

### 2. Object Detection
* **Endpoint:** `POST /api/detect`
* **Content-Type:** `multipart/form-data`
* **Query Parameters:**
  * `conf_threshold` *(float, optional)*: Override confidence limit (e.g. `0.4`).
  * `generate_audio` *(boolean, optional, default=true)*: Return base64 synthesized audio.
* **Multipart Form Fields:**
  * `file`: Binary image data (JPEG/PNG).
* **JSON Response Structure:**
  ```json
  {
    "success": true,
    "description": "The scene contains one blue bottle and two black keys.",
    "audio": "SUQzBAAAAAAA...", // Base64 encoded MP3 audio data
    "detections": [
      {
        "label": "bottle",
        "confidence": 0.895,
        "color": "blue",
        "box": [100, 150, 80, 200] // [x, y, width, height]
      },
      {
        "label": "key",
        "confidence": 0.762,
        "color": "black",
        "box": [300, 250, 40, 40]
      }
    ],
    "metrics": {
      "inference_time_ms": 42.15,
      "model_used": "yolov3-tiny",
      "image_width": 640,
      "image_height": 480
    }
  }
  ```

### Android Client Implementation (Retrofit Example)
To call this from Android, set up a Retrofit interface:

```interface
interface VisionService {
    @Multipart
    @POST("api/detect")
    fun detectObjects(
        @Part file: MultipartBody.Part,
        @Query("conf_threshold") conf: Float? = null,
        @Query("generate_audio") audio: Boolean = true
    ): Call<DetectionResponse>
}
```

You can then parse the `audio` Base64 string into a local file and play it via `MediaPlayer`, or directly pass the `description` string to Android's native `TextToSpeech` engine!
