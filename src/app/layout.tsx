import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroPlan · Light Mode",
  description: "Planner + Pomodoro + Week Plan",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-50 text-slate-900 antialiased">
        <div className="min-h-screen">
          <header className="border-b bg-white/70 backdrop-blur sticky top-0 z-10">
            <div className="mx-auto max-w-3xl px-4 py-4">
              <h1 className="text-2xl font-semibold tracking-tight">
                <span className="bg-gradient-to-r from-indigo-600 to-sky-500 bg-clip-text text-transparent">
                  NeuroPlan
                </span>{" "}
                · Light Mode
              </h1>
            </div>
          </header>
          <main className="mx-auto max-w-3xl px-4 py-6 space-y-6">{children}</main>
          <footer className="mx-auto max-w-3xl px-4 py-10 text-sm text-slate-500">
            MVP · Light Mode — EEG Plus coming soon
          </footer>
        </div>
      </body>
    </html>
  );
}
