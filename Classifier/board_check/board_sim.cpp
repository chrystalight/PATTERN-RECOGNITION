// board_sim.cpp - replays one recorded session (raw.csv) through the board's classifier, one fast row at a
// time, exactly as the firmware sees the sensors, and prints one CSV line per classified ride.
// Built and run by check_board.py:   board_sim <path/to/raw.csv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "../../ElevatorLogger_v2/classifier.h"

static ride::RideClassifier clf;   // ~115 KB of buffers: keep it off the stack

// split one line on commas, keeping empty fields (an F row's pressure is often empty)
static std::vector<std::string> split(const char* line) {
  std::vector<std::string> out(1);
  for (const char* p = line; *p && *p != '\n' && *p != '\r'; p++) {
    if (*p == ',') out.emplace_back();
    else out.back() += *p;
  }
  return out;
}

static float num(const std::string& s) { return s.empty() ? NAN : strtof(s.c_str(), nullptr); }

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: board_sim raw.csv\n"); return 1; }
  FILE* f = fopen(argv[1], "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

  printf("t_start_s,t_end_s,valid,building,launch_jerk_g_s,light_flicker_index,agree\n");
  std::vector<double> t;   // time of every fast sample (s from the first one), to report ride times
  char line[512];
  while (fgets(line, sizeof line, f)) {
    if (line[0] == '#') continue;
    // raw.csv row: host_epoch,F,t_us,ax_g,ay_g,az_g,light_raw,emf_raw,pres_Pa
    std::vector<std::string> a = split(line);
    if (a.size() != 9 || a[1] != "F" || a[2].empty()) continue;
    double t_us = atof(a[2].c_str());
    static double t0 = t_us;
    t.push_back((t_us - t0) / 1e6);

    float light = num(a[6]);
    if (clf.addSample(num(a[3]), num(a[4]), num(a[5]), (int16_t)(std::isnan(light) ? 0 : light), num(a[8]))) {
      const ride::Result& r = clf.result();
      if (r.valid)
        printf("%.2f,%.2f,1,%s,%.5f,%.5f,%d\n", t[r.startSample], t[r.endSample],
               knn::CLASS_NAMES[r.building], r.launchJerk, r.flicker, r.agree ? 1 : 0);
      else   // times = the barometer's ride span
        printf("%.2f,%.2f,0,\"%s\",,,\n", t[r.startSample], t[r.endSample], r.why);
    }
  }
  fclose(f);
  return 0;
}
