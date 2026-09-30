"""Causal, stationary feature construction.

Every function here obeys one rule (spec §0.4.1): the value produced for day
``t`` uses only data available at the close of day ``t``. No function in this
package shifts a series backwards, fits a scaler, or reads a quantile over the
whole sample. Scaling and winsorising are deliberately *not* here — they are
fitted objects, and live in :mod:`prism.features.scaling` where fit scope can
be enforced separately (spec §0.4.2).
"""
