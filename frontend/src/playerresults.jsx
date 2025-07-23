// playerresults.jsx
import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";

export default function PlayerResults() {
  const { videoId } = useParams();
  const [clips, setClips] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchClips = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/clips/${videoId}`);
        setClips(response.data);
      } catch (err) {
        console.error("無法獲取球員精華片段：", err);
      } finally {
        setLoading(false);
      }
    };
    fetchClips();
  }, [videoId]);

  if (loading) return <div className="text-center mt-8">載入中...</div>;

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold mb-4">球員精華片段</h1>
      {Object.entries(clips).length === 0 ? (
        <p>尚無可供下載的精華片段。</p>
      ) : (
        <div className="space-y-6">
          {Object.entries(clips).map(([number, files]) => (
            <div key={number} className="border rounded-lg p-4 shadow">
              <h2 className="text-lg font-semibold mb-2">球員 #{number}</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {files.map((file) => (
                  <video key={file} controls src={`http://localhost:8000/media/highlights/${videoId}/${file}`} className="w-full rounded-md" />
                ))}
              </div>
              <div className="mt-2">
                <button
                  onClick={() => files.forEach(f => window.open(`http://localhost:8000/media/highlights/${videoId}/${f}`, '_blank'))}
                  className="mt-2 px-4 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600"
                >
                  下載此球員所有精華片段
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
