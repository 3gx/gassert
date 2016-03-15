#include <cstdio>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __forceinline__
#endif

namespace printer {
enum {
  PRINT_F_MINUS          = (1 << 0),
  PRINT_F_PLUS           = (1 << 1),
  PRINT_F_SPACE          = (1 << 2),
  PRINT_F_NUM            = (1 << 3),
  PRINT_F_ZERO           = (1 << 4),
  PRINT_F_QUOTE          = (1 << 5),
  PRINT_F_UP             = (1 << 6),
  PRINT_F_UNSIGNED       = (1 << 7),
  PRINT_F_TYPE_G         = (1 << 8),
  PRINT_F_TYPE_E         = (1 << 9),
  MAX_CONVERT_LENGTH     = 43,
  OVERFLOW               = (1<<16),
  NaN                    = (1<<16)+1
};


__host__ __device__ __forceinline__
static bool ISNAN(double x) {return x != x; }

__host__ __device__ __forceinline__
static bool  ISINF(double x) { return x != 0.0 && x + x == x; }

__host__ __device__ __forceinline__
static void outchar(char *str, int &len, size_t &size, char ch)
{
  do
  {
    if (len + 1 < size)
      str[len] = ch;
    (len)++;
  } while (/* CONSTCOND */ 0);
}
__host__ __device__
static void printsep(char *str, int &len, size_t size)
{
		outchar(str, len, size, ',');
}


__host__ __device__
static int getexponent(double value)
{
	double tmp = (value >= 0.0) ? value : -value;
	int exponent = 0;

	/*
	 * We check for 99 > exponent > -99 in order to work around possible
	 * endless loops which could happen (at least) in the second loop (at
	 * least) if we're called with an infinite value.  However, we checked
	 * for infinity before calling this function using our ISINF() macro, so
	 * this might be somewhat paranoid.
	 */
	while (tmp < 1.0 && tmp > 0.0 && --exponent > -99)
		tmp *= 10;
	while (tmp >= 10.0 && ++exponent < 99)
		tmp /= 10;

	return exponent;
}

__host__ __device__
static double mypow10(int exponent)
{
	double result = 1;

	while (exponent > 0) {
		result *= 10;
		exponent--;
	}
	while (exponent < 0) {
		result /= 10;
		exponent++;
	}
	return result;
}

#define UINTMAX_MAX ((uint64_t)(-1))

__host__ __device__
static uint64_t cast(double value)
{
	uint64_t result;

	/*
	 * We check for ">=" and not for ">" because if UINTMAX_MAX cannot be
	 * represented exactly as an LDOUBLE value (but is less than LDBL_MAX),
	 * it may be increased to the nearest higher representable value for the
	 * comparison (cf. C99: 6.3.1.4, 2).  It might then equal the LDOUBLE
	 * value although converting the latter to UINTMAX_T would overflow.
	 */
	if (value >= UINTMAX_MAX)
		return UINTMAX_MAX;

	result = value;
	/*
	 * At least on NetBSD/sparc64 3.0.2 and 4.99.30, casting long double to
	 * an integer type converts e.g. 1.9 to 2 instead of 1 (which violates
	 * the standard).  Sigh.
	 */
	return (result <= value) ? result : result - 1;
}

__host__ __device__
static uint64_t myround(double value)
{
	double intpart = cast(value);

	return ((value -= intpart) < 0.5) ? intpart : intpart + 1;
}

__host__ __device__
static int getnumsep(int digits)
{
	int separators = (digits - ((digits % 3 == 0) ? 1 : 0)) / 3;
	return separators;
}


__host__ __device__
static int convert(uint64_t value, char *buf, size_t size, int base, int caps)
{
	const char *digits = caps ? "0123456789ABCDEF" : "0123456789abcdef";
	size_t pos = 0;

	/* We return an unterminated buffer with the digits in reverse order. */
	do {
		buf[pos++] = digits[value % base];
		value /= base;
	} while (value != 0 && pos < size);

	return (int)pos;
}
__host__ __device__ 
static int fmtflt(char *str, size_t size, double fvalue,
                   int width = 0, int precision = -1, int flags = 0)
{
  int len = 0;
	double ufvalue;
	uint64_t intpart;
	uint64_t fracpart;
	uint64_t mask;
	const char *infnan = NULL;
	char iconvert[MAX_CONVERT_LENGTH];
	char fconvert[MAX_CONVERT_LENGTH];
	char econvert[4];	/* "e-12" (without nul-termination). */
	char esign = 0;
	char sign = 0;
  
	int leadfraczeros = 0;
	int exponent = 0;
	int emitpoint = 0;
	int omitzeros = 0;
	int omitcount = 0;
	int padlen = 0;
	int epos = 0;
	int fpos = 0;
	int ipos = 0;
	int separators = (flags & PRINT_F_QUOTE);
	int estyle = (flags & PRINT_F_TYPE_E);

	/*
	 * AIX' man page says the default is 0, but C99 and at least Solaris'
	 * and NetBSD's man pages say the default is 6, and sprintf(3) on AIX
	 * defaults to 6.
	 */
	if (precision == -1)
		precision = 6;

	if (fvalue < 0.0)
		sign = '-';
	else if (flags & PRINT_F_PLUS)	/* Do a sign. */
		sign = '+';
	else if (flags & PRINT_F_SPACE)
		sign = ' ';

	if (ISNAN(fvalue))
		infnan = (flags & PRINT_F_UP) ? "NAN" : "nan";
	else if (ISINF(fvalue))
		infnan = (flags & PRINT_F_UP) ? "INF" : "inf";

	if (infnan != NULL) {
#if 0
		if (sign != 0)
			iconvert[ipos++] = sign;
		while (*infnan != '\0')
			iconvert[ipos++] = *infnan++;
		fmtstr(str, len, size, iconvert, width, ipos, flags);
#endif
		return len; /* NaN */
	}

	/* "%e" (or "%E") or "%g" (or "%G") conversion. */
	if (flags & PRINT_F_TYPE_E || flags & PRINT_F_TYPE_G) {
		if (flags & PRINT_F_TYPE_G) {
			/*
			 * For "%g" (and "%G") conversions, the precision
			 * specifies the number of significant digits, which
			 * includes the digits in the integer part.  The
			 * conversion will or will not be using "e-style" (like
			 * "%e" or "%E" conversions) depending on the precision
			 * and on the exponent.  However, the exponent can be
			 * affected by rounding the converted value, so we'll
			 * leave this decision for later.  Until then, we'll
			 * assume that we're going to do an "e-style" conversion
			 * (in order to get the exponent calculated).  For
			 * "e-style", the precision must be decremented by one.
			 */
			precision--;
			/*
			 * For "%g" (and "%G") conversions, trailing zeros are
			 * removed from the fractional portion of the result
			 * unless the "#" flag was specified.
			 */
			if (!(flags & PRINT_F_NUM))
				omitzeros = 1;
		}
		exponent = getexponent(fvalue);
		estyle = 1;
	}

again:
	/*
	 * Sorry, we only support 9, 19, or 38 digits (that is, the number of
	 * digits of the 32-bit, the 64-bit, or the 128-bit UINTMAX_MAX value
	 * minus one) past the decimal point due to our conversion method.
	 */
	switch (sizeof(uint64_t)) {
	case 16:
		if (precision > 38)
			precision = 38;
		break;
	case 8:
		if (precision > 19)
			precision = 19;
		break;
	default:
		if (precision > 9)
			precision = 9;
		break;
	}

	ufvalue = (fvalue >= 0.0) ? fvalue : -fvalue;
	if (estyle)	/* We want exactly one integer digit. */
		ufvalue /= mypow10(exponent);

	if ((intpart = cast(ufvalue)) == UINTMAX_MAX) {
    return 0; /* *overflow = 1 */
	}

	/*
	 * Factor of ten with the number of digits needed for the fractional
	 * part.  For example, if the precision is 3, the mask will be 1000.
	 */
	mask = mypow10(precision);
	/*
	 * We "cheat" by converting the fractional part to integer by
	 * multiplying by a factor of ten.
	 */
	if ((fracpart = myround(mask * (ufvalue - intpart))) >= mask) {
		/*
		 * For example, ufvalue = 2.99962, intpart = 2, and mask = 1000
		 * (because precision = 3).  Now, myround(1000 * 0.99962) will
		 * return 1000.  So, the integer part must be incremented by one
		 * and the fractional part must be set to zero.
		 */
		intpart++;
		fracpart = 0;
		if (estyle && intpart == 10) {
			/*
			 * The value was rounded up to ten, but we only want one
			 * integer digit if using "e-style".  So, the integer
			 * part must be set to one and the exponent must be
			 * incremented by one.
			 */
			intpart = 1;
			exponent++;
		}
	}

	/*
	 * Now that we know the real exponent, we can check whether or not to
	 * use "e-style" for "%g" (and "%G") conversions.  If we don't need
	 * "e-style", the precision must be adjusted and the integer and
	 * fractional parts must be recalculated from the original value.
	 *
	 * C99 says: "Let P equal the precision if nonzero, 6 if the precision
	 * is omitted, or 1 if the precision is zero.  Then, if a conversion
	 * with style `E' would have an exponent of X:
	 *
	 * - if P > X >= -4, the conversion is with style `f' (or `F') and
	 *   precision P - (X + 1).
	 *
	 * - otherwise, the conversion is with style `e' (or `E') and precision
	 *   P - 1." (7.19.6.1, 8)
	 *
	 * Note that we had decremented the precision by one.
	 */
	if (flags & PRINT_F_TYPE_G && estyle &&
	    precision + 1 > exponent && exponent >= -4) {
		precision -= exponent;
		estyle = 0;
		goto again;
	}

	if (estyle) {
		if (exponent < 0) {
			exponent = -exponent;
			esign = '-';
		} else
			esign = '+';

		/*
		 * Convert the exponent.  The sizeof(econvert) is 4.  So, the
		 * econvert buffer can hold e.g. "e+99" and "e-99".  We don't
		 * support an exponent which contains more than two digits.
		 * Therefore, the following stores are safe.
		 */
		epos = convert(exponent, econvert, 2, 10, 0);
		/*
		 * C99 says: "The exponent always contains at least two digits,
		 * and only as many more digits as necessary to represent the
		 * exponent." (7.19.6.1, 8)
		 */
		if (epos == 1)
			econvert[epos++] = '0';
		econvert[epos++] = esign;
		econvert[epos++] = (flags & PRINT_F_UP) ? 'E' : 'e';
	}

	/* Convert the integer part and the fractional part. */
	ipos = convert(intpart, iconvert, sizeof(iconvert), 10, 0);
	if (fracpart != 0)	/* convert() would return 1 if fracpart == 0. */
		fpos = convert(fracpart, fconvert, sizeof(fconvert), 10, 0);

	leadfraczeros = precision - fpos;

	if (omitzeros) {
		if (fpos > 0)	/* Omit trailing fractional part zeros. */
			while (omitcount < fpos && fconvert[omitcount] == '0')
				omitcount++;
		else {	/* The fractional part is zero, omit it completely. */
			omitcount = precision;
			leadfraczeros = 0;
		}
		precision -= omitcount;
	}

	/*
	 * Print a decimal point if either the fractional part is non-zero
	 * and/or the "#" flag was specified.
	 */
	if (precision > 0 || flags & PRINT_F_NUM)
		emitpoint = 1;
	if (separators)	/* Get the number of group separators we'll print. */
		separators = getnumsep(ipos);

	padlen = width                  /* Minimum field width. */
	    - ipos                      /* Number of integer digits. */
	    - epos                      /* Number of exponent characters. */
	    - precision                 /* Number of fractional digits. */
	    - separators                /* Number of group separators. */
	    - (emitpoint ? 1 : 0)       /* Will we print a decimal point? */
	    - ((sign != 0) ? 1 : 0);    /* Will we print a sign character? */

	if (padlen < 0)
		padlen = 0;

	/*
	 * C99 says: "If the `0' and `-' flags both appear, the `0' flag is
	 * ignored." (7.19.6.1, 6)
	 */
	if (flags & PRINT_F_MINUS)	/* Left justifty. */
		padlen = -padlen;
	else if (flags & PRINT_F_ZERO && padlen > 0) {
		if (sign != 0) {	/* Sign. */
			outchar(str, len, size, sign);
			sign = 0;
		}
		while (padlen > 0) {	/* Leading zeros. */
			outchar(str, len, size, '0');
			padlen--;
		}
	}
	while (padlen > 0) {	/* Leading spaces. */
		outchar(str, len, size, ' ');
		padlen--;
	}
	if (sign != 0)	/* Sign. */
		outchar(str, len, size, sign);
	while (ipos > 0) {	/* Integer part. */
		ipos--;
		outchar(str, len, size, iconvert[ipos]);
		if (separators > 0 && ipos > 0 && ipos % 3 == 0)
			printsep(str, len, size);
	}
	if (emitpoint) {	/* Decimal point. */
			outchar(str, len, size, '.');
	}
	while (leadfraczeros > 0) {	/* Leading fractional part zeros. */
		outchar(str, len, size, '0');
		leadfraczeros--;
	}
	while (fpos > omitcount) {	/* The remaining fractional part. */
		fpos--;
		outchar(str, len, size, fconvert[fpos]);
	}
	while (epos > 0) {	/* Exponent. */
		epos--;
		outchar(str, len, size, econvert[epos]);
	}
	while (padlen < 0) {	/* Trailing spaces. */
		outchar(str, len, size, ' ');
		padlen++;
	}

  return len;
} // void fmtflt(..)

} // namespace printer

int main(int argc, char * argv[])
{
  char str[256];
  printer::fmtflt(str, 256,  123.0f*argc);
  printf("%s \n", str);

}
