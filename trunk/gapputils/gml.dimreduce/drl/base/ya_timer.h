/****************************************************************************
 * ya_timer.h
 *    Shared stuff for doing program run times 
 *
 *
 * W. Michael Brown
 * 5/26/03
 ****************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_TIMEKEEPER
#define YA_TIMEKEEPER

#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <string>
#include <cstdlib>

#ifdef YA_MPI
#include "mpi.h"
#endif

//using namespace std;

#define YALANAMESPACE namespace yala {

YALANAMESPACE

// Center a string over the specified length
std::string ya_strcenter(const std::string &s, unsigned length) {
  std::string empty("                                                           ");
  unsigned half=length/2;
  unsigned spacer=half-(s.length()/2)-1;
  return std::string(spacer,' ')+s;
}

/// Keep track of cpu time required for operations
class YA_TimeKeeper {
 public:
  YA_TimeKeeper() {}
  ~YA_TimeKeeper() {}
  
  /// Set start time as now
  void start() { timestart = clock(); }
  /// Set end time as now
  void end() 
    { timeend = (clock()-timestart)/static_cast<double>(CLOCKS_PER_SEC); }

  #ifdef YA_MPI
  /// Set start time as now (mpi wall time)
  void mpi_wstart() { timestart = static_cast<clock_t>(MPI_Wtime()); }
  /// Set end time as now (mpi wall time)
  void mpi_wend() { timeend = MPI_Wtime()-timestart; }
  #endif

  friend ostream & operator<<(ostream &out, const YA_TimeKeeper &tk); 

  /// Return the number of CPU seconds timed
  double time() const
    { return timeend; }
    
  /// Return the number of CPU seconds from start() to now -without end() call
  double current_time() const 
    { return (clock()-timestart)/static_cast<double>(CLOCKS_PER_SEC); }

  /// Get the hours, minutes, and return seconds from total seconds
  double hms(const double total_seconds, int &hours, int &minutes) const {
    int int_seconds=static_cast<int>(total_seconds);
    double sfrac=total_seconds-int_seconds;
    div_t dt;
    dt=div(int_seconds,3600);
    hours=dt.quot;
    dt=div(dt.rem,60);
    minutes=dt.quot;
    return dt.rem+sfrac;
  }

 private:
  clock_t timestart;
  double timeend;
};

/// Visual Display of Operation Progress
class YA_ProgMeter {
 public:
  YA_ProgMeter() {}
  ~YA_ProgMeter() {}

  /// Start progress meter with a total and optionally output the percentage
  void start(std::string label, unsigned len, unsigned tot, bool perc) {
    length=len;
    total=tot;
    percent=perc;
    numbars=0;
    total_percent=0.0;
    iterate_percent=1.0/total;
    count_before_bar=static_cast<unsigned>(::ceil(static_cast<double>(total)/
                                                static_cast<double>(length)))-1;
    if (count_before_bar==0)
      count_before_bar=1;
    bar_frac_count=0;

    std::cerr << endl << ya_strcenter(label,length) << std::endl;
    for (unsigned i=0; i<length+2; i++)
      std::cerr << "-";

    std::cerr << "\n|" << std::flush;

    if (percent) {
      std::cerr << "  0.0%" << std::flush;
      std::cerr.setf(ios::fixed);
      std::cerr.setf(ios::showpoint);
      std::cerr.precision(1);
      std::cerr.setf(ios::right);
    }
  }    
  /// Advance the progress one more towards the total supplied in start()
  /** \note Thread safe for OpenMP **/
  void iterate() {
    #pragma omp critical
    {
    total_percent+=iterate_percent;
    bar_frac_count++;
    if (bar_frac_count>count_before_bar) {
      bar_frac_count=0;
      unsigned newbars=unsigned(total_percent*length)-numbars;
      numbars+=newbars;
      if (percent)
        std::cerr << "\b\b\b\b\b\b";
      for (unsigned i=0; i<newbars; i++)
        std::cerr << "#";
      std::cerr << std::flush;
      if (percent) {
        std::cerr << std::setw(5) << total_percent*100.0 << "%" << std::flush;
      }
    }
    }
  }
    
  /// Advance the progress n more towards the total supplied in start()
  /** \note Thread safe for OpenMP **/
  void iterate(unsigned n) {
    #pragma omp critical
    {
    total_percent+=iterate_percent*n;
    div_t answer=div(static_cast<int>(n+bar_frac_count),
                     static_cast<int>(count_before_bar));
    bar_frac_count=answer.rem;
    if (answer.quot>0) {
      unsigned newbars=unsigned(total_percent*length)-numbars;
      numbars+=newbars;
      if (percent)
        std::cerr << "\b\b\b\b\b\b";
      for (unsigned i=0; i<newbars; i++)
        std::cerr << "#";
      std::cerr << std::flush;
      if (percent) {
        std::cerr << std::setw(5) << total_percent*100.0 << "%" << std::flush;
      }
    }
    }
  }
    
  /// Close out the progress meter
  void finish() {
    if (percent)
      std::cerr << "\b\b\b\b\b\b";
    for (unsigned i=numbars; i<length; i++)
      std::cerr << "#";
    std::cerr << "|     \n";
    for (unsigned i=0; i<length+2; i++)
      std::cerr << "-";
    std::cerr << std::endl << std::flush;
  }
    
 private:
  unsigned total,length;
  double total_percent;
  double iterate_percent;
  unsigned count_before_bar;
  unsigned bar_frac_count;
  bool percent;

  unsigned numbars;
};

std::ostream & operator<<(std::ostream &out, const YA_TimeKeeper &tk) {
  double cpusec=tk.time();
  int hours, minutes;
  cpusec=tk.hms(cpusec,hours,minutes);
  
  if (hours>0)  
    out << hours << "h ";
  if (minutes>0)
    out << minutes << "m ";
  out << cpusec << "s";
  return out;
}

}
#endif
