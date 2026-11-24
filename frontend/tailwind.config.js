/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#070e22",
          900: "#0a1530",
          800: "#0f1f44",
          700: "#16295a",
          600: "#22397a",
          500: "#3a5190",
          300: "#8fa0c9",
          200: "#c3cde6",
          100: "#e3e9f6",
          50: "#f3f6fc",
        },
        brand: {
          700: "#1f46d6",
          600: "#2f5bff",
          500: "#4f77ff",
          400: "#7fa0ff",
          300: "#a9c0ff",
          100: "#e4ebff",
          50: "#f1f4ff",
        },
      },
      fontFamily: {
        display: ['"Bricolage Grotesque"', "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ["Figtree", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(10,21,48,.06), 0 4px 16px -6px rgba(10,21,48,.10)",
        pop: "0 12px 40px -12px rgba(10,21,48,.35)",
      },
      keyframes: {
        caret: { "0%,49%": { opacity: 1 }, "50%,100%": { opacity: 0 } },
        pop: { from: { opacity: 0, transform: "translateY(6px) scale(.98)" }, to: { opacity: 1, transform: "none" } },
      },
      animation: { caret: "caret 1s steps(1) infinite", pop: "pop .18s ease-out" },
    },
  },
  plugins: [],
};
