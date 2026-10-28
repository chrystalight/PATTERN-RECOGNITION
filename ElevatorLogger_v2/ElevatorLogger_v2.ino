#include <Wire.h>
#include <SPI.h>                              // SPI library
#include <Adafruit_Sensor.h>                  // Core sensor library
#include <Adafruit_ST7789.h>                  // Hardware-specific library for ST7789
#include <Adafruit_ADS1X15.h>                 // Hardware-specific library for ADS1x15
#include "Adafruit_BME680.h"                  // Hardware-specific library for BME680
#include <Adafruit_AHTX0.h>                   // Hardware-specific library for AHT20 - Connect external AHT20 with provided wire to Port J7
#include "SparkFun_BMA400_Arduino_Library.h"  // Hardware-specific library for BMA400
#include "esp_timer.h"                        // ESP32 hardware timer (fixed sample rate)

/*
  ELEVATOR LOGGER v2 - dual-rate data collection

  FAST stream (timer driven):  accelerometer, pressure, light, EMF
  SLOW stream (timer driven):  temperature, humidity (and gas, optional)
                               from the BME680, or from an AHT20 on port J7 if one is plugged in

  Nothing in the fast loop is allowed to block:
   - BME680 runs in forced mode with the gas heater OFF and is polled; a reading is picked up only once the
     sensor has finished, and the next one is started straight away.
   - AHT20 is triggered, then read back ~90 ms later (the Adafruit library would block for ~80 ms).

  Output over Serial is CSV with one record type per line (first column):
    F,t_us,ax_g,ay_g,az_g,light_raw,emf_raw,pres_Pa    fast row; pres_Pa is EMPTY unless a new pressure reading arrived
    S,t_us,temp_C,rh_pct,gas_kOhm                      slow row; gas is nan unless LOG_GAS = 1
    M,t_us,missed_ticks                                only appears if fast ticks were skipped (loop overran)
    # ...                                              comment / header lines
  t_us is microseconds since boot (64-bit, does not wrap).
*/

//------------------ SETTINGS ------------------
#define RUN_TIMING_TEST 0             // 1 = measure how long the work per fast tick takes, print results, then stop
                                      // 0 = run the logger
#define SERIAL_BAUD 1000000           // MUST match the baud rate of whatever is capturing the data

//FAST stream
#define FAST_PERIOD_US 5000UL         // PLACEHOLDER: fast tick period in us (5000 = 200 Hz). Use the timing test result.
#define BMA_ODR BMA400_ODR_200HZ      // accelerometer output data rate. Keep it equal to (or above) the fast rate.
#define BMA_RANGE BMA400_RANGE_2G     // +/-2 g gives the best resolution for elevator-scale accelerations
#define BMA_OSR BMA400_ACCEL_OSR_SETTING_3  // accelerometer oversampling: 3 = lowest noise (default 0 = lowest power)
#define BME_PRES_OS BME680_OS_4X      // pressure oversampling: lower = faster pressure updates, more noise
#define BME_TEMP_OS BME680_OS_1X      // temperature oversampling (needed for pressure compensation)
#define BME_HUM_OS BME680_OS_1X       // humidity oversampling. Measured in every BME680 cycle (adds ~2 ms per reading),
                                      // but only reported in the slow stream
#define ADS_DATA_RATE RATE_ADS1015_3300SPS  // ADC conversion rate (fastest for the ADS1015)

//SLOW stream
#define SLOW_PERIOD_US 1000000UL      // PLACEHOLDER: slow stream period in us (1,000,000 = 1 Hz). Keep > 100 ms.
#define AHT_CONVERSION_US 90000UL     // AHT20 needs >= 80 ms between trigger and readback
#define LOG_GAS 0                     // 1 = also log BME680 gas resistance. This turns the gas heater on, which makes
                                      // every BME680 reading much slower (pressure updates drop to a few Hz).

//Timing test
#define TIMING_TEST_MS 3000           // how long to run the fast-tick work back-to-back
#define TIMING_SAFETY_MARGIN 1.3f     // suggested period = worst-case tick time x this margin
//----------------------------------------------

//Timer tick counter: incremented by the hardware timer every FAST_PERIOD_US, cleared when a tick is serviced.
//Counting (rather than a single flag) lets us detect ticks that were missed because the loop ran long.
volatile uint32_t pendingTicks = 0;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;
void onFastTimer(void* arg) {
  portENTER_CRITICAL(&timerMux);
  pendingTicks++;
  portEXIT_CRITICAL(&timerMux);
}

//Pin Definitions
#define TFT_CS 38
#define TFT_RST -1
#define TFT_DC 48

#define LED_PIN 19
#define BUZZER_PIN 7

//Other Constants
#define BME_Addr 0x76
#define AHT_ADDR 0x38
#define BMA_ADDR 0x14
#define BMA_G_PER_LSB ((float)(2 << BMA_RANGE) / 2048.0f)  //12-bit data: +/-2 g range -> 2/2048 g per count
#define BME_TIMEOUT_US 200000         //if a BME680 reading hasn't finished after this long, collect it anyway

//Location in memory to store SPI object and display object
SPIClass* fspi = NULL;  //SPI
Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);

//Location In Memory to store Sensor Objects
Adafruit_BME680 bme;
Adafruit_AHTX0 aht;
Adafruit_ADS1015 ads;
BMA400 accelerometer;

//Latest values
float ax = 0, ay = 0, az = 0;       //g
int16_t rawLight = 0, rawEMF = 0;   //ADC counts (volts per count is printed in the log header)
float lastPres = NAN;               //Pa
float lastBmeT = NAN;               //deg C (BME680, used if no AHT20)
float lastBmeRH = NAN;              //% RH (BME680, used if no AHT20)
float lastGas = NAN;                //kOhm
bool pressureNew = false;           //true when lastPres has not yet been written to a fast row
uint32_t bmeReadCount = 0;          //number of completed BME680 readings (used by the timing test)

//State
bool AHTconnected = false;
int8_t bmaBeginErr = 0, bmaOdrErr = 0, bmaRangeErr = 0;  //BMA400 setup result codes (0 = OK)
bool bmeStarted = false;
int64_t bmeStartUs = 0;
bool ahtPending = false;
int64_t ahtTriggerUs = 0;
int64_t nextSlowUs = 0;

//------------------ TIMING HELPERS ------------------
//Timing statistics. Methods live inside the struct on purpose: the Arduino build auto-generates prototypes for
//free functions and puts them above the struct definition, which breaks any free function taking this type.
struct TimingStat {
  const char* name;
  uint32_t n;
  uint64_t sum;
  uint32_t minUs;
  uint32_t maxUs;

  void init(const char* label) {
    name = label;
    n = 0;
    sum = 0;
    minUs = 0xFFFFFFFF;
    maxUs = 0;
  }

  void add(uint32_t us) {
    n++;
    sum += us;
    if (us < minUs) minUs = us;
    if (us > maxUs) maxUs = us;
  }

  void print() const {
    if (n == 0) {
      Serial.printf("# %-36s %7s\n", name, "n/a");
      return;
    }
    Serial.printf("# %-36s %7lu %8lu %8lu %8lu\n", name, (unsigned long)n,
                  (unsigned long)minUs, (unsigned long)(sum / n), (unsigned long)maxUs);
  }
};

inline uint32_t nowUs() {
  return (uint32_t)esp_timer_get_time();  //unsigned subtraction of two nowUs() values is wrap-safe
}
//----------------------------------------------------

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.println("# Elevator Logger v2");

  //Init Buzzer and LED GPIO
  digitalWrite(LED_PIN, LOW);  //Turn off before setting to output to prevent flickering at boot
  digitalWrite(BUZZER_PIN, LOW);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  //Init Display (status text only - screen updates are far too slow to do inside the fast loop)
  fspi = new SPIClass(FSPI);
  fspi->begin(36, 37, 35);
  tft.init(135, 240);
  tft.setRotation(3);
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 50);
  tft.print("Elevator Logger");
  delay(1000);

  //Init all sensors
  Wire.begin(16, 15);       //I2C Bus #1 Init (For BME, BMA, ADC, AHT) - starts at 100 kHz
  scanI2C("100 kHz");       //expect 0x14 BMA400, 0x48 ADS1015, 0x76 BME680 (and 0x38 if an AHT20 is on J7)
  Wire.setClock(400000);    //400 kHz I2C: every sensor on the bus supports it, and it makes each read much shorter
  scanI2C("400 kHz");

  //Init BME680: pressure + temperature + humidity (IIR filter off, gas heater off unless LOG_GAS)
  if (!bme.begin(BME_Addr)) {
    Serial.println("# Could not find a valid BME680 sensor, check wiring!");
  }
  bme.setTemperatureOversampling(BME_TEMP_OS);
  bme.setHumidityOversampling(BME_HUM_OS);
  bme.setPressureOversampling(BME_PRES_OS);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_0);   //raw data; filter offline
#if LOG_GAS
  bme.setGasHeater(320, 150);  // 320*C for 150 ms
#else
  //Heater off. Adafruit_BME680 (v2.0.4) disables the heater when passed (0, 0) but keeps the old 150 ms duration
  //from begin() and still adds it to its "measurement ready" time, which capped pressure at ~5.7 Hz.
  //Setting a short duration first, then disabling, leaves that stored. 3 ms (not 1) gives a small safety margin,
  //because endReading() reports success even if the sensor has no new data yet (it would re-log the old value).
  bme.setGasHeater(320, 3);
  bme.setGasHeater(0, 0);      // heater off
#endif

  //Init AHT20 (library is used for init only; reads are done non-blocking below)
  if (!aht.begin()) {
    Serial.println("# No AHT20 on port J7 - temperature and humidity come from the BME680.");
    AHTconnected = false;
  } else {
    AHTconnected = true;
  }

  //Init ADC (ADS1015): EMF on A1, Light on A2
  if (!ads.begin()) {
    Serial.println("# Failed to initialize ADS.");
  }
  ads.setDataRate(ADS_DATA_RATE);

  //Init BMA400
  //Each step prints its result code (0 = OK, negative = error) so a failure is visible at boot
  bmaBeginErr = accelerometer.beginI2C(BMA400_I2C_ADDRESS_DEFAULT);
  bmaOdrErr = accelerometer.setODR(BMA_ODR);
  bmaRangeErr = accelerometer.setRange(BMA_RANGE);
  printBmaStatus();
  int8_t bmaOsrErr = accelerometer.setOSR(BMA_OSR);
  Serial.printf("# BMA400 setOSR:   %d%s\n", bmaOsrErr, bmaOsrErr == BMA400_OK ? " (OK)" : "  <-- ERROR");

#if RUN_TIMING_TEST
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(10, 50);
  tft.print("Timing test...");

  runTimingTest();

  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(10, 50);
  tft.print("Timing done");
  tft.setCursor(10, 80);
  tft.print("See Serial");
#else
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(10, 50);
  tft.print("Logging...");

  //Header (comment lines start with '#')
  Serial.println("# F,t_us,ax_g,ay_g,az_g,light_raw,emf_raw,pres_Pa");
  Serial.println("# S,t_us,temp_C,rh_pct,gas_kOhm");
  Serial.println("# M,t_us,missed_ticks");
  Serial.printf("# config fast_period_us=%lu slow_period_us=%lu baud=%d log_gas=%d aht=%d\n",
                (unsigned long)FAST_PERIOD_US, (unsigned long)SLOW_PERIOD_US, SERIAL_BAUD, LOG_GAS, AHTconnected);
  Serial.printf("# adc_volts_per_count=%.6f\n", ads.computeVolts(1));

  nextSlowUs = esp_timer_get_time();

  esp_timer_create_args_t timerArgs = {};
  timerArgs.callback = &onFastTimer;
  timerArgs.name = "fast";
  esp_timer_handle_t fastTimer;
  esp_timer_create(&timerArgs, &fastTimer);
  esp_timer_start_periodic(fastTimer, FAST_PERIOD_US);
#endif
}

void loop() {
#if RUN_TIMING_TEST
  delay(1000);  //test already finished in setup(); set RUN_TIMING_TEST to 0 to log
#else
  if (pendingTicks == 0) return;  //wait for the timer

  //Grab and clear the tick count. More than one tick means we fell behind and skipped some.
  uint32_t ticks;
  portENTER_CRITICAL(&timerMux);
  ticks = pendingTicks;
  pendingTicks = 0;
  portEXIT_CRITICAL(&timerMux);

  int64_t t = esp_timer_get_time();
  if (ticks > 1) {
    Serial.printf("M,%lld,%lu\n", (long long)t, (unsigned long)(ticks - 1));
  }
  fastTick(t);
  slowTick(t);
#endif
}

void printBmaStatus() {
  Serial.printf("# BMA400 beginI2C: %d%s\n", bmaBeginErr, bmaBeginErr == BMA400_OK ? " (OK)" : "  <-- ERROR: not detected at 0x14");
  Serial.printf("# BMA400 setODR:   %d%s\n", bmaOdrErr, bmaOdrErr == BMA400_OK ? " (OK)" : "  <-- ERROR");
  Serial.printf("# BMA400 setRange: %d%s\n", bmaRangeErr, bmaRangeErr == BMA400_OK ? " (OK)" : "  <-- ERROR");
}

//Lists every I2C address that acknowledges, so a missing sensor is obvious at boot
void scanI2C(const char* label) {
  Serial.printf("# I2C scan (%s):", label);
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) Serial.printf(" 0x%02X", addr);
  }
  Serial.println();
}

//------------------ FAST STREAM ------------------
//Reads the 6 acceleration bytes straight from the BMA400 in one I2C transaction (~4x faster than the library's
//getSensorData, which also re-reads the range setting every call). If the read fails, logs nan.
void readAccel() {
  uint8_t d[6];
  Wire.beginTransmission(BMA_ADDR);
  Wire.write(0x04);  //ACC_X_LSB; X, Y, Z follow as LSB/MSB pairs
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom((uint8_t)BMA_ADDR, (size_t)6) != 6) {
    ax = ay = az = NAN;
    return;
  }
  for (int i = 0; i < 6; i++) d[i] = Wire.read();
  int16_t raw[3];
  for (int i = 0; i < 3; i++) {
    raw[i] = (((int16_t)d[2 * i + 1] << 8) | d[2 * i]) & 0x0FFF;  //12-bit two's complement
    if (raw[i] > 2047) raw[i] -= 4096;
  }
  ax = raw[0] * BMA_G_PER_LSB;
  ay = raw[1] * BMA_G_PER_LSB;
  az = raw[2] * BMA_G_PER_LSB;
}

void readAds() {
  rawEMF = ads.readADC_SingleEnded(1);
  rawLight = ads.readADC_SingleEnded(2);
}

//Checks whether the BME680 has finished its forced-mode measurement: the mode bits in ctrl_meas (0x74)
//drop back to 00 (sleep) when it's done. (Don't read the status register 0x1D here: reading it clears the
//new-data flag, so the library then thinks there's no data and stalls.)
bool bmeDataReady() {
  Wire.beginTransmission(BME_Addr);
  Wire.write(0x74);  //ctrl_meas
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom((uint8_t)BME_Addr, (size_t)1) != 1) return false;
  return (Wire.read() & 0x03) == 0;
}

//Non-blocking BME680 handling. Never waits: if the sensor is still measuring, it returns immediately.
//The chip's own status bit is checked before endReading(), because if the data isn't ready the Bosch driver
//inside the library retries with 10 ms delays (up to ~60 ms), which would stall the fast loop.
void pollBme() {
  if (!bmeStarted) {
    bmeStarted = (bme.beginReading() != 0);
    bmeStartUs = esp_timer_get_time();
    return;
  }
  if (bme.remainingReadingMillis() > 0) return;  //library's estimate: still measuring
  bool timedOut = (esp_timer_get_time() - bmeStartUs) > BME_TIMEOUT_US;
  if (!bmeDataReady() && !timedOut) return;      //chip says not done yet: try again next tick
  if (timedOut) Serial.println("# BME680 reading timed out");
  if (bme.endReading()) {
    lastPres = bme.pressure;                //Pa
    lastBmeT = bme.temperature;             //deg C
    lastBmeRH = bme.humidity;               //% RH
#if LOG_GAS
    lastGas = bme.gas_resistance / 1000.0;  //kOhm
#endif
    pressureNew = true;
    bmeReadCount++;
  }
  bmeStarted = (bme.beginReading() != 0);   //start the next reading right away
  bmeStartUs = esp_timer_get_time();
}

//Writes one F row. Returns the number of bytes written.
int emitFastRow(int64_t t) {
  char line[96];
  int n = snprintf(line, sizeof(line), "F,%lld,%.4f,%.4f,%.4f,%d,%d,",
                   (long long)t, ax, ay, az, rawLight, rawEMF);
  if (pressureNew) {
    n += snprintf(line + n, sizeof(line) - n, "%.2f", lastPres);
    pressureNew = false;
  }
  line[n++] = '\n';
  Serial.write((const uint8_t*)line, n);
  return n;
}

void fastTick(int64_t t) {
  readAccel();   //first, so it lines up best with the timestamp
  readAds();
  pollBme();
  emitFastRow(t);
}

//------------------ SLOW STREAM ------------------
void ahtTrigger() {
  Wire.beginTransmission(AHT_ADDR);
  Wire.write(0xAC);  //trigger measurement
  Wire.write(0x33);
  Wire.write(0x00);
  ahtPending = (Wire.endTransmission() == 0);
  ahtTriggerUs = esp_timer_get_time();
}

//Reads back a finished AHT20 measurement. Returns false if the data is not ready or the read failed.
bool ahtRead(float& tC, float& rh) {
  uint8_t d[7];
  if (Wire.requestFrom((uint8_t)AHT_ADDR, (size_t)7) != 7) return false;
  for (int i = 0; i < 7; i++) d[i] = Wire.read();
  if (d[0] & 0x80) return false;  //still busy
  uint32_t rawH = ((uint32_t)d[1] << 12) | ((uint32_t)d[2] << 4) | (d[3] >> 4);
  uint32_t rawT = ((uint32_t)(d[3] & 0x0F) << 16) | ((uint32_t)d[4] << 8) | d[5];
  rh = rawH * 100.0f / 1048576.0f;
  tC = rawT * 200.0f / 1048576.0f - 50.0f;
  return true;
}

void emitSlowRow(int64_t t, float tC, float rh) {
  Serial.printf("S,%lld,%.2f,%.2f,%.2f\n", (long long)t, tC, rh, lastGas);
}

void slowTick(int64_t t) {
  if (ahtPending) {
    if (t - ahtTriggerUs >= (int64_t)AHT_CONVERSION_US) {
      float tC, rh;
      if (ahtRead(tC, rh)) {
        ahtPending = false;
        emitSlowRow(t, tC, rh);
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));  //heartbeat: toggles once per slow row
      } else if (t - ahtTriggerUs > (int64_t)(3 * AHT_CONVERSION_US)) {
        ahtPending = false;  //give up on this one, try again next slow period
      }
    }
  } else if (t >= nextSlowUs) {
    nextSlowUs = t + SLOW_PERIOD_US;
    if (AHTconnected) {
      ahtTrigger();
    } else {
      emitSlowRow(t, lastBmeT, lastBmeRH);
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    }
  }
}

//------------------ TIMING TEST ------------------
//Runs the fast-tick work back-to-back (no waiting for the timer) and reports how long each part takes,
//then suggests the fastest safe FAST_PERIOD_US.
void runTimingTest() {
  TimingStat sAccel, sAds, sBme, sRow, sTick, sAhtTrig, sAhtRead;
  sAccel.init("BMA400 getSensorData");
  sAds.init("ADS1015 read EMF + Light");
  sBme.init("BME680 pollBme (all calls)");
  sRow.init("F row format + Serial write");
  sTick.init("TOTAL fast tick");
  sAhtTrig.init("AHT20 trigger");
  sAhtRead.init("AHT20 readback + S row");

  Serial.printf("# Timing test: fast-tick work back-to-back for %d ms (F rows below are test output)\n", TIMING_TEST_MS);

  uint64_t lineBytes = 0;
  bmeReadCount = 0;
  uint32_t testStart = nowUs();
  while ((nowUs() - testStart) < (uint32_t)TIMING_TEST_MS * 1000UL) {
    int64_t t = esp_timer_get_time();
    uint32_t tickStart = nowUs();
    uint32_t t0;

    t0 = nowUs();
    readAccel();
    sAccel.add(nowUs() - t0);

    t0 = nowUs();
    readAds();
    sAds.add(nowUs() - t0);

    t0 = nowUs();
    pollBme();
    sBme.add(nowUs() - t0);

    t0 = nowUs();
    int n = emitFastRow(t);
    sRow.add(nowUs() - t0);

    sTick.add(nowUs() - tickStart);
    lineBytes += n;
  }
  uint32_t elapsedUs = nowUs() - testStart;

  //AHT20 cost (trigger and readback happen inside a fast tick once per slow period)
  if (AHTconnected) {
    for (int i = 0; i < 10; i++) {
      uint32_t t0 = nowUs();
      ahtTrigger();
      sAhtTrig.add(nowUs() - t0);
      delay(AHT_CONVERSION_US / 1000 + 10);
      float tC, rh;
      t0 = nowUs();
      if (ahtRead(tC, rh)) emitSlowRow(esp_timer_get_time(), tC, rh);
      sAhtRead.add(nowUs() - t0);
      ahtPending = false;
    }
  }

  Serial.println("#");
  Serial.println("# ---------------- TIMING RESULTS (microseconds) ----------------");
  Serial.printf("# %-36s %7s %8s %8s %8s\n", "Step", "n", "min", "avg", "max");
  sAccel.print();
  sAds.print();
  sBme.print();
  sRow.print();
  sTick.print();
  sAhtTrig.print();
  sAhtRead.print();
  Serial.println("# ---------------------------------------------------------------");

  //Worst case: a tick that also carries the AHT20 trigger or readback
  uint32_t ahtTrigMax = sAhtTrig.n ? sAhtTrig.maxUs : 0;
  uint32_t ahtReadMax = sAhtRead.n ? sAhtRead.maxUs : 0;
  uint32_t ahtExtra = (ahtTrigMax > ahtReadMax) ? ahtTrigMax : ahtReadMax;
  uint32_t worstUs = sTick.maxUs + ahtExtra;
  uint32_t suggestedUs = ((uint32_t)(worstUs * TIMING_SAFETY_MARGIN) / 500 + 1) * 500;  //round up to 0.5 ms
  float avgLine = (float)lineBytes / sTick.n;
  float suggestedHz = 1e6f / suggestedUs;
  float linkLoad = avgLine * suggestedHz / (SERIAL_BAUD / 10.0f) * 100.0f;

  Serial.printf("# Worst-case fast tick (incl. AHT20 work): %lu us\n", (unsigned long)worstUs);
  Serial.printf("# Fastest possible fast rate (no margin):  %.0f Hz\n", 1e6f / worstUs);
  Serial.printf("# Suggested: #define FAST_PERIOD_US %lu   (%.0f Hz, %.0f%% margin)\n",
                (unsigned long)suggestedUs, suggestedHz, (TIMING_SAFETY_MARGIN - 1.0f) * 100.0f);
  Serial.printf("# BME680 pressure updates: %.1f Hz at the current oversampling/heater settings (sensor limit)\n",
                bmeReadCount * 1e6f / elapsedUs);
  Serial.printf("# Serial load at the suggested rate: %.0f%% of %d baud (avg F row %.0f bytes). Keep this well under 100%%.\n",
                linkLoad, SERIAL_BAUD, avgLine);
  Serial.println("# Set BMA_ODR to match the chosen fast rate (200 Hz -> BMA400_ODR_200HZ, 400 Hz -> BMA400_ODR_400HZ).");
  Serial.println("# Then set RUN_TIMING_TEST to 0 and update FAST_PERIOD_US.");

  //Device check repeated here so it is visible without scrolling back to the boot messages
  Serial.println("# ---------------- DEVICE CHECK ----------------");
  Wire.setClock(100000);
  scanI2C("100 kHz");
  Wire.setClock(400000);
  scanI2C("400 kHz");
  Serial.println("# Expected: 0x14 BMA400, 0x48 ADS1015, 0x76 BME680 (0x38 only if an AHT20 is on J7)");
  printBmaStatus();
  Serial.printf("# BMA400 getSensorData now: %d%s\n", accelerometer.getSensorData(),
                "  (0 = OK, negative = error)");
  Serial.printf("# Accel check, library: %.4f %.4f %.4f g\n",
                accelerometer.data.accelX, accelerometer.data.accelY, accelerometer.data.accelZ);
  readAccel();
  Serial.printf("# Accel check, direct:  %.4f %.4f %.4f g  (should match the library line within ~0.01 g)\n", ax, ay, az);              
                
}