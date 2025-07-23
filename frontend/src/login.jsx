import React from 'react';
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';

export default function Login() {
  const [status, setStatus] = useState("");
  const navigate = useNavigate();

  const handleLoginSuccess = async (credentialResponse) => {
    const token = credentialResponse.credential;
    const res = await fetch("http://localhost:8000/auth/google", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });

    const result = await res.json();
    if (result.status === "success") {
      localStorage.setItem("userEmail", result.email);
      setStatus(`歡迎, ${result.name}`);
      navigate("/upload");
    } else {
      setStatus("登入失敗");
    }
  };

  return (
    <GoogleOAuthProvider clientId="579826086181-9r8n60lfa2erse942e2i1ngp8d5gh7hs.apps.googleusercontent.com">
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-gray-100 to-gray-200 px-4">
        <div className="bg-white shadow-xl rounded-xl p-10 w-full max-w-md text-center">
          <h1 className="text-2xl font-semibold text-gray-800 mb-3">自動精華影片系統</h1>
          <p className="text-gray-600 text-sm mb-6">請使用 Google 帳號登入以開始使用</p>
          
          <div className="flex justify-center">
            <div className="w-[250px]"> {/* 控制按鈕寬度 */}
              <GoogleLogin
                onSuccess={handleLoginSuccess}
                onError={() => setStatus("登入失敗")}
                theme="outline"
                size="large"
                text="signin_with"
              />
            </div>
          </div>

          {status && <p className="text-sm text-gray-700 mt-4">{status}</p>}
        </div>
      </div>
    </GoogleOAuthProvider>
  );
}
