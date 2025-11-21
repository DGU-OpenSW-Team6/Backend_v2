import { useState, useEffect } from "react";
import axios from "axios";

const BACKEND = "https://sketchcheck.shop";

function App() {
  const [jwt, setJwt] = useState<string | null>(localStorage.getItem("jwt"));
  const [result, setResult] = useState<any>(null);
  const [file, setFile] = useState<File | null>(null);

  // axios 요청 시 JWT 자동 삽입
  axios.interceptors.request.use((config) => {
    const token = localStorage.getItem("jwt");
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  });

  // /callback?token=... 처리
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");

    if (token) {
      localStorage.setItem("jwt", token);
      setJwt(token);
      setResult({ login: "success", token });
      window.history.replaceState({}, "", "/");
    }
  }, []);

  // Google 로그인 시작
  const startGoogleLogin = () => {
    const googleURL =
      `https://accounts.google.com/o/oauth2/v2/auth?` +
      `client_id=133050396922-856c4qtiu21ta4h3s9j2tbua02kk1c23.apps.googleusercontent.com` +
      `&redirect_uri=https://sketchcheck.shop/auth/callback` +
      `&response_type=code&scope=openid%20email%20profile`;

    window.location.href = googleURL;
  };

  // 일반 API 호출
  const callApi = async (path: string) => {
    try {
      const res = await axios.get(`${BACKEND}${path}`);
      setResult(res.data);
    } catch (err: any) {
      setResult(err.response?.data || err);
    }
  };

  // 업로드
  const uploadImage = async () => {
    if (!file) return alert("이미지 먼저 선택하세요.");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${BACKEND}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err: any) {
      setResult(err.response?.data || err);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Backend v2 전체 API 테스트 페이지</h1>

      <p>
        <b>JWT:</b> {jwt ? jwt.substring(0, 25) + "..." : "없음"}
      </p>

      <button onClick={startGoogleLogin}>Google 로그인</button>
      <br />
      <br />

      {/* 기본 API */}
      <button onClick={() => callApi("/")}>GET /</button>
      <button onClick={() => callApi("/returnScore")}>GET /returnScore</button>
      <br />
      <br />

      {/* 인증 필요 */}
      <button onClick={() => callApi("/mypage")}>GET /mypage</button>
      <button onClick={() => callApi("/myuploads")}>GET /myuploads</button>
      <br />
      <br />

      {/* 업로드 */}
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      <button onClick={uploadImage}>POST /upload</button>

      <hr />

      <h3>API 결과</h3>
      <pre
        style={{
          background: "#f0f0f0",
          padding: "20px",
          whiteSpace: "pre-wrap",
          overflowX: "auto",
          color: "#222",
          border: "1px solid #bbb",
          borderRadius: "8px",
          fontFamily: "Consolas, monospace",
          fontSize: "14px",
        }}
      >
        {result ? JSON.stringify(result, null, 2) : "API 호출 결과 없음"}
      </pre>
    </div>
  );
}

export default App;
