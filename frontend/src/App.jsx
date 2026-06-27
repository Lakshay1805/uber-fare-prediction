import { useState } from 'react'
import './App.css'

const DEFAULT_FORM = {
  pickup_latitude: '',
  pickup_longitude: '',
  dropoff_latitude: '',
  dropoff_longitude: '',
  passenger_count: '',
  pickup_datetime: '',
}

function App() {
  const [form, setForm] = useState(DEFAULT_FORM)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setResult(null)
    setError(null)
    setLoading(true)

    const payload = {
      pickup_latitude: parseFloat(form.pickup_latitude),
      pickup_longitude: parseFloat(form.pickup_longitude),
      dropoff_latitude: parseFloat(form.dropoff_latitude),
      dropoff_longitude: parseFloat(form.dropoff_longitude),
      passenger_count: parseInt(form.passenger_count, 10),
      pickup_datetime: form.pickup_datetime.replace('T', ' '),
    }

    try {
      const res = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}`)
      }

      const data = await res.json()
      setResult(data.predicted_fare)
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const isFormValid =
    form.pickup_latitude &&
    form.pickup_longitude &&
    form.dropoff_latitude &&
    form.dropoff_longitude &&
    form.passenger_count &&
    form.pickup_datetime

  return (
    <div className="app">
      <header className="header">
        <div className="header-icon">U</div>
        <h1>
          Uber <span>Fare Predictor</span>
        </h1>
      </header>

      <main className="main">
        <div className="card">
          <div className="card-header">
            <h2>Predict Your Fare</h2>
            <p>Enter ride details to get an estimated fare amount.</p>
          </div>

          <form className="card-body" onSubmit={handleSubmit}>
            <div className="form-grid">

              <div className="form-group">
                <label htmlFor="pickup_latitude">Pickup Latitude</label>
                <input
                  id="pickup_latitude"
                  name="pickup_latitude"
                  type="number"
                  step="any"
                  placeholder="40.7"
                  value={form.pickup_latitude}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="pickup_longitude">Pickup Longitude</label>
                <input
                  id="pickup_longitude"
                  name="pickup_longitude"
                  type="number"
                  step="any"
                  placeholder="-73.9"
                  value={form.pickup_longitude}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="dropoff_latitude">Dropoff Latitude</label>
                <input
                  id="dropoff_latitude"
                  name="dropoff_latitude"
                  type="number"
                  step="any"
                  placeholder="40.8"
                  value={form.dropoff_latitude}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="dropoff_longitude">Dropoff Longitude</label>
                <input
                  id="dropoff_longitude"
                  name="dropoff_longitude"
                  type="number"
                  step="any"
                  placeholder="-73.8"
                  value={form.dropoff_longitude}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="passenger_count">Passengers</label>
                <input
                  id="passenger_count"
                  name="passenger_count"
                  type="number"
                  min="1"
                  max="10"
                  placeholder="2"
                  value={form.passenger_count}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="pickup_datetime">Pickup Date & Time</label>
                <input
                  id="pickup_datetime"
                  name="pickup_datetime"
                  type="datetime-local"
                  value={form.pickup_datetime}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              className="submit-btn"
              disabled={!isFormValid || loading}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Predicting…
                </>
              ) : (
                'Predict Fare'
              )}
            </button>

            {result !== null && (
              <div className="result success">
                <div className="result-label">Estimated Fare</div>
                <div className="result-value">
                  <span className="dollar">$</span>
                  {result.toFixed(2)}
                </div>
              </div>
            )}


            {error && (
              <div className="result error">
                <div className="result-label">Error</div>
                <div className="result-value">{error}</div>
              </div>
            )}
          </form>
        </div>
      </main>

      <footer className="footer">
        Uber Fare Prediction · Powered by ML
      </footer>
    </div>
  )
}

export default App
