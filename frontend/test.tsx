import { useEffect } from "react";

export default function TestBackend() {
  useEffect(() => {
    fetch("/api/test")
      .then((res) => res.json())
      .then((data) => console.log("Backend response:", data))
      .catch(console.error);
  }, []);

  return <div>Check console for backend response!</div>;
}
