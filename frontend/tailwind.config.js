/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        'gemini-dark': '#0f0f0f',
        'gemini-input': '#1e1e1e',
      }
    },
  },
  plugins: [],
}