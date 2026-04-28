/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        medical: "0 12px 40px rgba(30, 64, 175, 0.08)",
      },
    },
  },
  plugins: [],
};
