"use client";
import Link from "next/link";
import { useEffect, useState } from "react";

const EEG_URL = process.env.NEXT_PUBLIC_EEG_URL ?? "http://localhost:8501";

export default function EEGPage() {
  const [status, setStatus] = useState<"loading" | "connected" | "offline">("loading");
  const [immersive, setImmersive] = useState(true); // 默认沉浸式

  async function checkConnection() {
    try {
      await fetch(EEG_URL, { mode: "no-cors" });
      setStatus("connected");
    } catch {
      setStatus("offline");
    }
  }
  useEffect(() => {
    checkConnection();
    const id = setInterval(checkConnection, 10000);
    return () => clearInterval(id);
  }, []);

  return (
    <main className={`w-full ${immersive ? "p-0" : "p-6"} mx-auto`}>
      {/* 顶部栏（沉浸式时缩小为细条） */}
      <div
        className={`w-full sticky top-0 z-10 backdrop-blur-sm ${
          immersive ? "px-3 py-2" : "px-6 py-3"
        } border-b bg-white/70`}
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${
                status === "connected"
                  ? "bg-green-500"
                  : status === "offline"
                  ? "bg-red-500"
                  : "bg-yellow-400"
              }`}
              title={status}
            />
            <span className="text-sm opacity-70 hidden sm:block">
              {status === "connected" ? "EEG Connected" : status === "offline" ? "EEG Offline" : "Checking…"}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setImmersive((v) => !v)}
              className="text-xs sm:text-sm rounded-md border px-3 py-1 hover:bg-gray-50"
              title="Toggle immersive mode"
            >
              {immersive ? "Exit Immersive" : "Enter Immersive"}
            </button>
            <a
              href={EEG_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs sm:text-sm rounded-md bg-blue-600 px-3 py-1 text-white hover:bg-blue-700"
            >
              Open in New Tab
            </a>
            {!immersive && (
              <Link href="/" className="text-xs sm:text-sm rounded-md border px-3 py-1 hover:bg-gray-50">
                ← Back to Light Mode
              </Link>
            )}
          </div>
        </div>
      </div>

      {/* 标题（非沉浸式时显示） */}
      {!immersive && (
        <div className="max-w-7xl mx-auto px-6 mt-4">
          <h1 className="text-2xl font-semibold mb-1">🧠 EEG Mode (Experimental)</h1>
          <p className="text-sm opacity-80 mb-4">
            Connect Muse to view real-time focus (β/α) & memory ((θ+γ)/α) metrics.
          </p>
        </div>
      )}

      {/* 主区域：全屏 iframe / 状态提示 */}
      <div className={`${immersive ? "" : "max-w-7xl mx-auto px-6"}`}>
        {status !== "connected" ? (
          <div className="p-6 border rounded-xl bg-gray-50 text-center text-gray-600 mt-4">
            {status === "loading" ? (
              <p>Checking EEG server…</p>
            ) : (
              <>
                <p className="font-medium mb-2">⚠️ EEG server not running</p>
                <p className="text-sm mb-3">Start Streamlit, this page will auto-detect:</p>
                <pre className="bg-gray-900 text-green-400 p-3 rounded-md text-xs text-left overflow-auto">
{`cd ~/Neuroplan/src/app/eeg
streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false`}
                </pre>
              </>
            )}
          </div>
        ) : (
          <iframe
            src={EEG_URL}
            className={`w-full rounded-none sm:rounded-xl border-0 sm:border shadow-none sm:shadow-lg`}
            style={{
              // 沉浸式：占满整个视窗；非沉浸式：留出上下间距
              height: immersive ? "calc(100vh - 56px)" : "calc(100vh - 220px)",
            }}
            referrerPolicy="no-referrer"
          />
        )}
      </div>
    </main>
  );
}
