import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

function Results() {
  const [videos, setVideos] = useState({});
  const [highlights, setHighlights] = useState({});
  const navigate = useNavigate();

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await axios.get("http://localhost:8000/videos");
        setVideos(res.data);

        // 同步抓已完成影片的球員 highlight
        for (const [id, info] of Object.entries(res.data)) {
          if (info.status === "completed" && !highlights[id]) {
            const highlightRes = await axios.get(
              `http://localhost:8000/highlights/${id}`
            );
            setHighlights((prev) => ({ ...prev, [id]: highlightRes.data.players }));
          }
        }
      } catch (err) {
        console.error("Failed to fetch videos:", err);
      }
    };

    fetchStatus(); // 初始執行一次
    const interval = setInterval(fetchStatus, 5000); // 每5秒更新
    return () => clearInterval(interval);
  }, [highlights]);

  const groupVideos = () => {
    const pending = [];
    const processing = [];
    const completed = [];

    for (const [id, info] of Object.entries(videos)) {
      const entry = { id, ...info };
      if (info.status === "pending") pending.push(entry);
      else if (info.status === "processing") processing.push(entry);
      else if (info.status === "completed") completed.push(entry);
    }

    return { pending, processing, completed };
  };

  const { pending, processing, completed } = groupVideos();

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-4">影片處理狀態</h1>

      <button
        onClick={() => navigate("/")}
        className="mb-6 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        回到上傳頁面
      </button>

      <section className="mb-6">
        <h2 className="text-xl font-semibold">待處理</h2>
        {pending.length > 0 ? (
          <ul className="list-disc ml-6">
            {pending.map((v) => (
              <li key={v.id}>{v.filename}</li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-500 ml-6">無待處理影片</p>
        )}
      </section>

      <section className="mb-6">
        <h2 className="text-xl font-semibold">處理中</h2>
        {processing.length > 0 ? (
          <ul className="list-disc ml-6">
            {processing.map((v) => (
              <li key={v.id}>{v.filename}</li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-500 ml-6">無處理中影片</p>
        )}
      </section>

      <section>
        <h2 className="text-xl font-semibold">處理完成</h2>
        {completed.length > 0 ? (
          completed.map((v) => (
            <div key={v.id} className="border p-4 mb-4 rounded">
              <p className="font-medium">{v.filename}</p>
              <p className="text-gray-500 mb-2">影片已處理完成，可下載球員精華片段：</p>
              {highlights[v.id] ? (
                <ul className="ml-4">
                  {highlights[v.id]
                    .sort((a, b) => parseInt(a.player) - parseInt(b.player))
                    .map((h) => (
                      <li key={h.file} className="mb-1">
                        球員 {h.player}：
                        <a
                          href={`http://localhost:8000${h.file}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 underline ml-1"
                          download
                        >
                          下載精華片段
                        </a>
                      </li>
                    ))}
                </ul>
              ) : (
                <p className="text-gray-400">載入球員資料中...</p>
              )}
            </div>
          ))
        ) : (
          <p className="text-gray-500 ml-6">無完成影片</p>
        )}
      </section>
    </div>
  );
}

export default Results;
