import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'WebShop Fine-tuning Agent',
  description:
    'AI agent for the WebShop benchmark with trajectory recording for fine-tuning',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
