import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api/client"; // assumes src/api/client.js exports axios instance
import "../styles/signin.css";

export default function SignIn() {
  const navigate = useNavigate();

  // Controlled inputs for sign-in
  const [signInEmail, setSignInEmail] = useState("");
  const [signInPassword, setSignInPassword] = useState("");
  const [signInLoading, setSignInLoading] = useState(false);
  const [signInError, setSignInError] = useState("");

  // Controlled inputs for sign-up
  const [signUpName, setSignUpName] = useState("");
  const [signUpEmail, setSignUpEmail] = useState("");
  const [signUpPassword, setSignUpPassword] = useState("");
  const [signUpConfirm, setSignUpConfirm] = useState("");
  const [signUpLoading, setSignUpLoading] = useState(false);
  const [signUpError, setSignUpError] = useState("");

  // ref for container to toggle signup / signin view
  const contRef = useRef(null);

  useEffect(() => {
    const el = contRef.current;
    // preserve original behaviour by listening to clicks on .img-btn element (which exists in DOM)
    const imgBtn = el?.querySelector(".img-btn");
    const handler = () => {
      el?.classList.toggle("s-signup");
    };
    imgBtn?.addEventListener("click", handler);
    return () => imgBtn?.removeEventListener("click", handler);
  }, []);

  // Basic client-side validation helpers
  function _validateEmail(email) {
    return /^\S+@\S+\.\S+$/.test(email);
  }

  // Login handler
  async function handleSignIn(e) {
    e.preventDefault();
    setSignInError("");
    if (!signInEmail || !signInPassword) {
      setSignInError("Please enter email and password.");
      return;
    }
    if (!_validateEmail(signInEmail)) {
      setSignInError("Please enter a valid email address.");
      return;
    }

    setSignInLoading(true);
    try {
      // Adjust endpoint if your backend login route differs
      const res = await api.post("/api/auth/login", {
        email: signInEmail,
        password: signInPassword,
      });

      // Expecting { token: "...", user: {...} } or similar
      const data = res.data || {};
      const token = data.token || data.access_token || null;
      const user = data.user || data;

      if (token) {
        localStorage.setItem("auth_token", token);
      }
      if (user) {
        localStorage.setItem("user", JSON.stringify(user));
      }

      // on success, navigate to upload page (adjust as needed)
      navigate("/upload");
    } catch (err) {
      // Try read backend message
      const msg =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        "Login failed";
      setSignInError(String(msg));
    } finally {
      setSignInLoading(false);
    }
  }

  // Sign-up handler
  async function handleSignUp(e) {
    e.preventDefault();
    setSignUpError("");

    if (!signUpName || !signUpEmail || !signUpPassword || !signUpConfirm) {
      setSignUpError("All fields are required.");
      return;
    }
    if (!_validateEmail(signUpEmail)) {
      setSignUpError("Please enter a valid email address.");
      return;
    }
    if (signUpPassword !== signUpConfirm) {
      setSignUpError("Passwords do not match.");
      return;
    }
    if (signUpPassword.length < 6) {
      setSignUpError("Password must be at least 6 characters.");
      return;
    }

    setSignUpLoading(true);
    try {
      // Adjust endpoint if your backend signup route differs
      const res = await api.post("/api/auth/signup", {
        name: signUpName,
        email: signUpEmail,
        password: signUpPassword,
      });

      const data = res.data || {};
      // If backend returns token on signup, store it
      const token = data.token || data.access_token || null;
      const user = data.user || data;

      if (token) {
        localStorage.setItem("auth_token", token);
      }
      if (user) {
        localStorage.setItem("user", JSON.stringify(user));
      }

      // navigate to upload or sign-in view
      // If token present -> logged in; else switch to sign-in panel to let user login
      if (token) {
        navigate("/upload");
      } else {
        // toggle back to sign-in view to allow user to login
        contRef.current?.classList.remove("s-signup");
        alert("Sign up successful. Please sign in.");
      }
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        "Sign up failed";
      setSignUpError(String(msg));
    } finally {
      setSignUpLoading(false);
    }
  }

  return (
    <div id="signin-root">
      <div className="floating-shapes">
        <div className="shape"></div>
        <div className="shape"></div>
        <div className="shape"></div>
        <div className="shape"></div>
      </div>

      <div className="cont" ref={contRef}>
        {/* Sign In Form */}
        <div className="form sign-in">
          <h2>Sign In</h2>
          <label>
            <span>Email Address</span>
            <input
              type="email"
              name="email"
              value={signInEmail}
              onChange={(e) => setSignInEmail(e.target.value)}
            />
          </label>
          <label>
            <span>Password</span>
            <input
              type="password"
              name="password"
              value={signInPassword}
              onChange={(e) => setSignInPassword(e.target.value)}
            />
          </label>

          {signInError && <div className="form-error">{signInError}</div>}

          <button
            className="submit"
            type="button"
            onClick={handleSignIn}
            disabled={signInLoading}
          >
            {signInLoading ? "Signing in…" : "Sign In"}
          </button>
        </div>

        <div className="sub-cont">
          <div className="img">
            <div className="img-text m-up">
              <h1>New here?</h1>
              <p>Sign up and discover</p>
            </div>
            <div className="img-text m-in">
              <h1>One of us?</h1>
              <p>Just sign in</p>
            </div>
            <div className="img-btn">
              <span className="m-up">Sign Up</span>
              <span className="m-in">Sign In</span>
            </div>
          </div>

          {/* Sign Up Form */}
          <div className="form sign-up">
            <h2>Sign Up</h2>

            <label>
              <span>Name</span>
              <input
                name="name"
                type="text"
                value={signUpName}
                onChange={(e) => setSignUpName(e.target.value)}
              />
            </label>

            <label>
              <span>Email</span>
              <input
                name="email"
                type="email"
                value={signUpEmail}
                onChange={(e) => setSignUpEmail(e.target.value)}
              />
            </label>

            <label>
              <span>Password</span>
              <input
                name="password"
                type="password"
                value={signUpPassword}
                onChange={(e) => setSignUpPassword(e.target.value)}
              />
            </label>

            <label>
              <span>Confirm Password</span>
              <input
                name="confirm-password"
                type="password"
                value={signUpConfirm}
                onChange={(e) => setSignUpConfirm(e.target.value)}
              />
            </label>

            {signUpError && <div className="form-error">{signUpError}</div>}

            <button
              type="button"
              className="submit"
              onClick={handleSignUp}
              disabled={signUpLoading}
            >
              {signUpLoading ? "Signing up…" : "Sign Up Now"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
