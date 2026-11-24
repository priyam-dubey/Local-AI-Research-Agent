/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: { 950: '#070a1c', 900: '#0b0f2a', 800: '#111638', 700: '#1a2050', 600: '#252c63' },
        accent: { DEFAULT: '#6366f1', soft: '#818cf8', strong: '#4f46e5' },
      },
      fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'Roboto', 'sans-serif'] },
      keyframes: { fadeIn: { from: { opacity: 0, transform: 'translateY(4px)' }, to: { opacity: 1, transform: 'none' } } },
      animation: { 'fade-in': 'fadeIn .25s ease-out' },
    },
  },
  plugins: [],
}
