import React from "react";
import { GoogleLogin } from "@react-oauth/google";
import { jwtDecode } from "jwt-decode";
import { useAuth } from "./auth/AuthProvider.jsx";
import { useLocation, useNavigate } from "react-router-dom";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const loc = useLocation();
  const from = (loc.state && loc.state.from) || "/";

  return (
    <div className="min-h-[100vh] grid place-items-center">
      <div className="bg-white dark:bg-neutral-900 border dark: borde-neutral-800 rounded-x2 p-6 w-[360px] max-w-[92vw]">
        <h1 className="text-lg font-bold mb-2 text-neutral-900 dark:text-netural-100">五人制足球自動剪輯精華系統</h1>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">使用 Google 帳戶登入以上傳與查看影片</p>
        <GoogleLogin
          onSuccess={(credentialResponse) => {
            try {
              const payload = jwtDecode(credentialResponse.credential);
              const profile = {
                name: payload.name,
                email: payload.email,
                picture: payload.picture,
                sub: payload.sub,
              };
              login(profile);
              navigate(from, { replace: true });
            } catch (e) {
              alert("無法解析登入資訊：" + e.message);
            }
          }}
          onError={() => {
            alert("登入失敗，請重試。");
          }}
          useOneTap
        />
      </div>
    </div>
  );
}
