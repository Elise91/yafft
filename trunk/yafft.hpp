#ifndef _YAFFT_HPP_
#define _YAFFT_HPP_

#include <complex>

#include <boost/shared_ptr.hpp>

#include "yafft_simd.hpp"
#include "yafft_twiddles.hpp"
#include "yafft_unswizzle.hpp"

namespace yafft {

  template <typename FP_T>
  class iyafft_windowfn {
  public:
    virtual ~iyafft_windowfn() {}
    virtual FP_T * const get_window(const size_t windowsz) = 0;
  };

  enum window_t {
    HAMMING,
    HANN,
    GAUSS,
    BLACKMAN
  };

  template <typename FP_T>
  boost::shared_ptr<iyafft_windowfn<FP_T> > get_windowfn(const window_t wt);

  typedef iyafft_windowfn<float> iyafft_float_windowfn_t;
  typedef boost::shared_ptr<iyafft_float_windowfn_t> iyafft_float_windowfn_ptr;

  typedef iyafft_windowfn<double> iyafft_double_windowfn_t;
  typedef boost::shared_ptr<iyafft_double_windowfn_t> iyafft_double_windowfn_ptr;

  template <typename FP_T>
  class iyafft {
  public:
    virtual ~iyafft() {}
    
    virtual bool init() = 0;
    
    virtual void run_forward(std::complex<FP_T> * const,
			     std::complex<FP_T> * const) = 0;
    
    virtual void run_forward(FP_T * const,
			     std::complex<FP_T> * const) = 0;
    
    virtual void run_reverse(std::complex<FP_T> * const,
			     std::complex<FP_T> * const) = 0;
    
    virtual void run_reverse(std::complex<FP_T> * const,
			     FP_T * const) = 0;
  };

  template <typename FP_T>
  boost::shared_ptr<iyafft<FP_T> > get_fft(const size_t N);
  
  typedef boost::shared_ptr<iyafft<float> > iyafft_float_ptr;
  typedef boost::shared_ptr<iyafft<double> > iyafft_double_ptr;

  template <typename FP_T>
  class iyafft_pow2 : public iyafft<FP_T> {
  public:

    virtual ~iyafft_pow2() {}
    
    virtual void run_forward_nounswizzle(std::complex<FP_T> * const) = 0;
    
    virtual void run_forward_mul_noout(std::complex<FP_T> * const, 
				       FP_T * const) = 0;
    
    virtual void run_reverse_mul_noout(std::complex<FP_T> * const, 
				       FP_T * const) = 0;
    
    virtual void run_reverse_mul_noin(FP_T * const,
				      FP_T * const,
				      std::complex<FP_T> * const) = 0;    

    virtual void run_forward_mul_noin(FP_T * const,
				      FP_T * const,
				      std::complex<FP_T> * const) = 0;    

    virtual FP_T * const get_work() = 0;
  };
}

#endif //  _YAFFT_HPP_
