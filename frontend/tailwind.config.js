/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.06), 0 10px 30px -10px rgba(0,0,0,0.6)",
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};
