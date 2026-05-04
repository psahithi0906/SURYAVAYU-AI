import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders VASUDHA AI dashboard', () => {
  render(<App />);
  expect(screen.getByText(/Renewable generation forecasts/i)).toBeInTheDocument();
});
