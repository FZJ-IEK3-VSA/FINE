# Rolling Horizon TODO

## Teilweise erledigt: IP-abhaengige Parameter mit Stock-Jahren

In `fine/expansionModules/rollingHorizon.py` filtert `_filterComponentParametersForInterval` generische Dict-Parameter aktuell teils mit `rollingHorizonYears + stockYears`.

Das ist fuer stock-/decommissioning-relevante Parameter sinnvoll, z.B. `materialIntensity`, weil Material-Recovery die Intensitaet des urspruenglichen Kommissionierungsjahres braucht.

Fuer operative Tail-Parameter ist das aber falsch bzw. kann spaetere RH-/Reoptimierungsfenster brechen. Beispiele:

- `operationRateMax`
- `operationRateMin`
- `operationRateFix`
- `commodityCostTimeSeries`
- `commodityRevenueTimeSeries`
- `materialCollection`

Diese Parameter sollten im Tail nur auf die aktiven Optimierungsjahre gefiltert werden, nicht auf alte `stockYears`.

Kontext: Das wurde beim Nachdenken ueber Event-/Disruption-Reoptimierung sichtbar.

Status:

- Fuer die neue Reoptimization-Logik ist das Issue geloest: `_filterComponentParametersForReoptimization` trennt operative Parameter von stock-relevanten Parametern.
- Fuer die klassische Rolling-Horizon-Logik ist das Issue noch offen: `_filterComponentParametersForInterval` nutzt fuer generische Dict-Parameter weiterhin `rollingHorizonYears + stockYears`.

Naechster Schritt:

- `_filterComponentParametersForInterval` analog zur Reoptimization-Filterlogik ueberarbeiten.
- Danach mit einem IP-abhaengigen `operationRateMax`-Fall testen, z.B. Materialimport-Disruption.
