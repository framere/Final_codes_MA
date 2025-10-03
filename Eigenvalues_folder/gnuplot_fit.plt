# Configuration
set terminal wxt enhanced size 800,600
set logscale xy
set xlabel "Index" font ",12"
set ylabel "|Eigenvalue|" font ",12"
set title "Eigenvalues Spectrum (Log-Log) - First N-600" font ",14"
set grid
set key left top

# File and title lists
files = "uracil_eigenvalues.dat formaldehyde_eigenvalues.dat H2_eigenvalues.dat"
titles = "Uracil Formaldehyde H2"

# Get N from each file and set cutoffs
stats "uracil_eigenvalues.dat" nooutput
N_uracil = STATS_records
last_uracil = N_uracil - 600

stats "formaldehyde_eigenvalues.dat" nooutput
N_formaldehyde = STATS_records
last_formaldehyde = N_formaldehyde - 650

stats "H2_eigenvalues.dat" nooutput
N_H2 = STATS_records
last_H2 = N_H2 - 200

print sprintf("Uracil: %d points, plotting first: %d", N_uracil, last_uracil)
print sprintf("Formaldehyde: %d points, plotting first: %d", N_formaldehyde, last_formaldehyde)
print sprintf("H2: %d points, plotting first: %d", N_H2, last_H2)

# Define colors for better distinction
set style line 1 lc rgb '#e41a1c' pt 7 ps 0.5
set style line 2 lc rgb '#377eb8' pt 7 ps 0.5
set style line 3 lc rgb '#4daf4a' pt 7 ps 0.5

# Plot data with different cutoffs for each file (excluding first 300)
plot "uracil_eigenvalues.dat" every ::300::last_uracil using 1:(abs($2)) \
     with points linestyle 1 title sprintf("Uracil (N=%d)", last_uracil-299), \
     "formaldehyde_eigenvalues.dat" every ::300::last_formaldehyde using 1:(abs($2)) \
     with points linestyle 2 title sprintf("Formaldehyde (N=%d)", last_formaldehyde-299), \
     "H2_eigenvalues.dat" every ::300::last_H2 using 1:(abs($2)) \
     with points linestyle 3 title sprintf("H2 (N=%d)", last_H2-299)

# Define fitting function: model(x) = p1 * exp(-p2 * x^p3)
f1(x) = p1_1 * exp(-p1_2 * x**p1_3)
f2(x) = p2_1 * exp(-p2_2 * x**p2_3)
f3(x) = p3_1 * exp(-p3_2 * x**p3_3)

# Initial guesses - use values from first data point
p1_1 = 1000; p1_2 = 0.00001; p1_3 = 1.0
p2_1 = 1000; p2_2 = 0.00001; p2_3 = 1.0
p3_1 = 1000; p3_2 = 0.00001; p3_3 = 1.0

# Set fitting parameters
set fit maxiter 200
set fit limit 1e-8

# Fit each dataset (excluding first 300)
fit f1(x) "uracil_eigenvalues.dat" every ::300::last_uracil using 1:(abs($2)) via p1_1, p1_2, p1_3
fit f2(x) "formaldehyde_eigenvalues.dat" every ::300::last_formaldehyde using 1:(abs($2)) via p2_1, p2_2, p2_3
fit f3(x) "H2_eigenvalues.dat" every ::300::last_H2 using 1:(abs($2)) via p3_1, p3_2, p3_3

# Replot with fits
replot f1(x) with lines lc rgb '#e41a1c' lw 2 dashtype 2 title sprintf("Uracil fit: %.2e*exp(-%.2e*x^{%.2f})", p1_1, p1_2, p1_3), \
       f2(x) with lines lc rgb '#377eb8' lw 2 dashtype 2 title sprintf("Formaldehyde fit: %.2e*exp(-%.2e*x^{%.2f})", p2_1, p2_2, p2_3), \
       f3(x) with lines lc rgb '#4daf4a' lw 2 dashtype 2 title sprintf("H2 fit: %.2e*exp(-%.2e*x^{%.2f})", p3_1, p3_2, p3_3)

pause -1 "Press Enter to save and continue..."

# Save as PNG with high quality
set terminal pngcairo enhanced size 1200,800 font "Arial,12"
set output "eigenvalues_spectrum.png"
replot
set output

# Save as PDF (vector format, better for publications)
set terminal pdfcairo enhanced size 6,4.5 font "Arial,12"
set output "eigenvalues_spectrum.pdf"
replot
set output

# Return to interactive mode
set terminal wxt enhanced size 800,600
print "Plots saved as:"
print "  - eigenvalues_spectrum.png"
print "  - eigenvalues_spectrum.pdf"