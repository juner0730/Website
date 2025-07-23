import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

export default function App() {
  const [video, setVideo] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setVideo(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!video) return;
    setUploading(true);
    setMessage("正在上傳中...");

    const formData = new FormData();
    formData.append("video", video);

    try {
      const res = await axios.post("http://localhost:8000/upload", formData);
      setMessage(res.data.message);
      setTimeout(() => navigate("/results"), 1000);
    } catch (err) {
      console.error(err);
      setMessage("上傳失敗");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-gray-100 to-gray-200 px-4">
      <div className="bg-white shadow-xl rounded-xl p-10 w-full max-w-md text-center">
        <h1 className="text-2xl font-semibold text-gray-800 mb-4">影片上傳系統</h1>
        <p className="text-gray-600 text-sm mb-6">請選擇影片並上傳進行處理</p>

        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="block w-full mb-4 text-sm text-gray-600
                     file:mr-4 file:py-2 file:px-4
                     file:rounded file:border-0
                     file:text-sm file:font-semibold
                     file:bg-blue-100 file:text-blue-700
                     hover:file:bg-blue-200"
        />

        <button
          onClick={handleUpload}
          disabled={uploading}
          className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600 transition disabled:opacity-50"
        >
          {uploading ? "上傳中..." : "上傳"}
        </button>

        {message && <p className="mt-4 text-gray-700">{message}</p>}
      </div>
    </div>
  );
}
