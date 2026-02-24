// src/app/page.tsx
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-neutral-50 p-8">
      {/* 标题 */}
      <div className="mb-12 text-center">
        <h1 className="mb-4 text-5xl font-bold text-gray-900">NeuroPlan 🧠</h1>
        <p className="text-xl text-gray-600">
          Optimize your study workflow with Neuroscience & AI
        </p>
      </div>

      {/* 卡片容器 */}
      <div className="grid w-full max-w-4xl grid-cols-1 gap-8 md:grid-cols-2">
        {/* Light Mode Card - 点击去往你刚才做好的页面 */}
        <Link
          href="/light"
          className="group block transform rounded-2xl border border-gray-200 bg-white p-8 shadow-sm transition-all hover:-translate-y-1 hover:shadow-md"
        >
          <div className="mb-4 text-4xl">☀️</div>
          <h2 className="mb-3 text-2xl font-bold text-blue-600 group-hover:text-blue-700">
            Light Mode
          </h2>
          <p className="mb-4 font-medium text-gray-900">Behavioral Optimization</p>
          <ul className="space-y-2 text-sm text-gray-500">
            <li>• Smart Task Decomposition</li>
            <li>• AI Chat Planner</li>
            <li>• Week View Schedule</li>
          </ul>
        </Link>

        {/* EEG Mode Card - 点击去往你的 EEG 页面 */}
        <Link
          href="/eeg"
          className="group block transform rounded-2xl border border-gray-200 bg-white p-8 shadow-sm transition-all hover:-translate-y-1 hover:shadow-md"
        >
          <div className="mb-4 text-4xl">⚡</div>
          <h2 className="mb-3 text-2xl font-bold text-purple-600 group-hover:text-purple-700">
            EEG Mode
          </h2>
          <p className="mb-4 font-medium text-gray-900">Brainwave Optimization</p>
          <ul className="space-y-2 text-sm text-gray-500">
            <li>• Real-time Focus Tracking</li>
            <li>• Attention & Memory States</li>
            <li>• Closed-loop Feedback</li>
          </ul>
        </Link>
      </div>

      <footer className="mt-16 text-sm text-gray-400">
        MVP v0.1 · Powered by Next.js & Muse
      </footer>
    </div>
  );
}
