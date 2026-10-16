import { Component, type ReactNode } from 'react';

/** Contains a rendering failure to one region and says what failed. */
export class ErrorBoundary extends Component<
  { label: string; children: ReactNode; resetKey?: unknown },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidUpdate(prev: { resetKey?: unknown }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="ws-error" role="alert">
        <p>
          {this.props.label} could not be displayed: {this.state.error.message}
        </p>
      </div>
    );
  }
}
