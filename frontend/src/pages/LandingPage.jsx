import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Landingpage.css';

export default function LandingPage() {
  const navigate = useNavigate();

  useEffect(() => {
    const cursor = document.getElementById('cursor');
    function onMove(e) {
      if (cursor) {
        cursor.style.left = `${e.clientX - 10}px`;
        cursor.style.top = `${e.clientY - 10}px`;
      }
    }
    document.addEventListener('mousemove', onMove);

    function onScroll() {
      const navbar = document.getElementById('navbar');
      if (!navbar) return;
      if (window.scrollY > 50) navbar.classList.add('scrolled');
      else navbar.classList.remove('scrolled');
    }
    window.addEventListener('scroll', onScroll);

    const observerOptions = { threshold: 0.1, rootMargin: '0px 0px -50px 0px' };
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => { if (entry.isIntersecting) entry.target.classList.add('visible'); });
    }, observerOptions);
    document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

    const style = document.createElement('style');
    style.textContent = `
      @keyframes particle {
        0% { transform: scale(0) translateY(0); opacity: 1; }
        100% { transform: scale(1) translateY(-100px); opacity: 0; }
      }
    `;
    document.head.appendChild(style);

    const hoverEls = document.querySelectorAll('.btn-primary, .btn-secondary, .step-circle-3d, .feature-card-3d');
    function onEnter() { const c = document.getElementById('cursor'); if (c) { c.style.transform = 'scale(2)'; c.style.background = 'rgba(0, 255, 255, 0.3)'; } }
    function onLeave() { const c = document.getElementById('cursor'); if (c) { c.style.transform = 'scale(1)'; c.style.background = 'transparent'; } }
    hoverEls.forEach(el => { el.addEventListener('mouseenter', onEnter); el.addEventListener('mouseleave', onLeave); });

    return () => {
      document.removeEventListener('mousemove', onMove);
      window.removeEventListener('scroll', onScroll);
      style.remove();
      hoverEls.forEach(el => { el.removeEventListener('mouseenter', onEnter); el.removeEventListener('mouseleave', onLeave); });
      observer.disconnect();
    };
  }, []);

  // All primary CTAs should route to signin
  const goToSignIn = () => navigate('/signin');
  const goToUpload = () => {
  navigate("/upload");
};

  function createParticles() {
    for (let i = 0; i < 50; i++) {
      const particle = document.createElement('div');
      particle.style.cssText = `position: fixed; width: 4px; height: 4px; background: #00ffff; border-radius: 50%; pointer-events: none; z-index: 10000; left: ${Math.random()*100}vw; top: ${Math.random()*100}vh; animation: particle 2s ease-out forwards;`;
      document.body.appendChild(particle);
      setTimeout(() => particle.remove(), 2000);
    }
  }

  // Optional: keep the visual particle effect then navigate
  function startDemoAndNavigate() {
    createParticles();
    // small delay so particles are visible before navigating
    setTimeout(() => goToUpload(), 300);
  }

  return (
    <div>
      <div className="cursor" id="cursor" aria-hidden="true" />

      <nav className="navbar" id="navbar" role="navigation" aria-label="Main navigation">
        <div className="nav-container">
          <div className="logo">CODE2PAPER</div>
          <div className="nav-links">
            {/* internal anchors that scroll to sections on the same page */}
            <a href="#features">Features</a>
            <a href="#demo">How it Works</a>


            {/* CTA uses navigate to /signin */}
            <button
              className="cta-nav"
              onClick={goToUpload}
              aria-label="Get started - go to sign in"
              type="button"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      <section className="hero" aria-labelledby="hero-heading">
        <div className="hero-bg" id="heroBg" aria-hidden="true" />
        <div className="floating-shapes" aria-hidden="true">
          <div className="shape" />
          <div className="shape" />
          <div className="shape" />
          <div className="shape" />
        </div>

        <div className="hero-content">
          <h1 id="hero-heading" className="hero-title">
            TRANSFORM MACHINE LEARNING CODE INTO RESEARCH PAPERS
          </h1>

          <p className="hero-subtitle">
            Revolutionary AI-powered platform that automatically generates comprehensive research papers from your machine learning code.
            Join the future of academic writing.
          </p>

          <div className="hero-cta">
            {/* Navigate directly to sign-in */}
            <button
              className="btn-primary"
              onClick={goToUpload}
              aria-label="Start creating - go to sign in"
              type="button"
            >
              🔥 Start Creating
            </button>


            
          </div>
        </div>
      </section>

      <section className="features-3d" id="features">
        <div className="features-container">
          <h2 className="section-title fade-in">NEXT-GEN FEATURES</h2>
          <div className="features-grid-3d">
            {[
              {icon:'⚙️', title:'Automated Notebook Parsing', text:'Transforms your Jupyter notebooks into structured research insights by extracting methods, results, and visual data automatically.'},
              {icon:'⚡', title:'AI-Powered Paper Generation', text:'Generate complete research papers from code in minutes including Abstract, Introduction, Methodology, and Results sections.'},
              {icon:'🧠', title:'Smart Citation Management', text:'Automatically detects citations, enriches them from local or external databases, and formats them in IEEE style.'},
              {icon:'🎯', title:'Context-Aware Title & Summary', text:'Creates concise, descriptive titles and abstracts aligned with your notebook’s core contribution.'},
              {icon:'🧩', title:'Selective Section Generation', text:'Choose specific sections to generate like Abstract, Methodology, or Results for faster and more focused paper creation.'},
              {icon:'💾 ', title:'Multi-Format Export', text:'Export your generated paper in .docx and .pdf formats.'}
            ].map((f, idx) => (
              <div className="feature-card-3d fade-in" key={idx}>
                <span className="feature-icon-3d" aria-hidden="true">{f.icon}</span>
                <h3>{f.title}</h3>
                <p>{f.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="demo-3d" id="demo">
        <div className="demo-container">
          <h2 className="section-title fade-in">HOW THE MAGIC HAPPENS</h2>
          <div className="workflow-3d">
            <div className="workflow-line" aria-hidden="true" />
            {[1,2,3,4].map(i => (
              <div className="workflow-step-3d fade-in" key={i}>
                <div className="step-circle-3d" aria-hidden="true">{i}</div>
                <h4>{['Upload Notebook','AI Understanding','Generate Paper','Export & Share'][i-1]}</h4>
                <p>{['Drag & drop your .ipynb file','Smart analysis of code, results & logic','Create all or selected sections automatically','Get a ready-to-edit .docx or .pdf instantly'][i-1]}</p>
              </div>
            ))}
          </div>

          <button
            className="btn-primary"
            style={{ marginTop: '4rem' }}
            onClick={goToUpload}
            aria-label="Experience the future - go to sign in"
            type="button"
          >
            🚀 Experience the Future
          </button>
        </div>
      </section>

      <footer className="footer-3d" role="contentinfo">
        <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '1.1rem' }}>© 2025 CODE2PAPER • Revolutionizing Academic Publishing with AI</p>
      </footer>
    </div>
  );
}
