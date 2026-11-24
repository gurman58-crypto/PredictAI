import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ToastProvider } from "./context/Toast";
import { ModelsProvider } from "./context/Models";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <ModelsProvider>
          <App />
        </ModelsProvider>
      </ToastProvider>
    </BrowserRouter>
  </React.StrictMode>
);
