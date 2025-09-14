import { useAuth } from "../auth/AuthProvider.jsx";
import { GoogleLogin } from "@react-oauth/google";
import jwtDecode from "jwt-decode";

export default function Topbar() {
  const { user, login } = useAuth();

  return (
    <header className="h-16 sticky top-0 z-30 bg-white/70 backdrop-blur-sm border-b border-neutral-200">
      <div className="max-w-6xl h-full mx-auto px-6 flex items-center justify-between">
        <div className="font-medium tracking-wide">自動剪輯系統</div>
        <div>
          {!user ? (
            <GoogleLogin
              onSuccess={(cred) => {
                // 解析 Google Credential JWT 取使用者名稱/信箱頭像
                const payload = jwtDecode(cred.credential);
                login({
                  name: payload.name,
                  email: payload.email,
                  picture: payload.picture,
                  sub: payload.sub,
                });
              }}
              onError={() => console.warn("Google Login failed")}
              useOneTap
              text="signin_with"
              locale="zh-TW"
            />
          ) : (
            <div className="flex items-center gap-3">
              <img
                src={user.picture}
                alt=""
                className="w-8 h-8 rounded-full border border-neutral-200"
              />
              <span className="text-sm text-neutral-700">
                歡迎回來 {user.name.split(" ")[0]}
              </span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
