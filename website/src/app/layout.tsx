import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title:
    "Physics-Guided Machine Learning for Non-Newtonian Rheology | Research Monograph by MA RIZWAN",
  description:
    "Conference presentation and interactive research platform: Physics-Guided Machine Learning for Non-Newtonian Polymer Flow (Xanthan Gum & Carboxymethyl Cellulose). Chemical Engineering Minor Project.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark scroll-smooth antialiased">
      <body className="min-h-screen bg-[#09090b] text-[#f4f4f5] selection:bg-zinc-200 selection:text-zinc-950 font-sans overflow-x-hidden">
        {children}
      </body>
    </html>
  );
}
