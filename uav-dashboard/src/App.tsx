import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { TelemetryProvider } from './context/TelemetryContext';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import Settings from './pages/Settings';
import BiometricDetection from './pages/BiometricDetection';
import AudioDistress from './pages/AudioDistress';

function App() {
  return (
    <Router>
      <TelemetryProvider>
        <div className="min-h-screen bg-gray-100">
          <Header />
          <main className="py-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/history" element={<History />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/biometrics" element={<BiometricDetection />} />
              <Route path="/audio-distress" element={<AudioDistress />} />
            </Routes>
          </main>
        </div>
      </TelemetryProvider>
    </Router>
  );
}

export default App;