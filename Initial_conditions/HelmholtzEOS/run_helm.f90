      program run_helm
      include 'implno.dek'
      include 'vector_eos.dek'

! Batch, non-interactive Helmholtz EOS driver (Timmes & Swesty 2000).
! stdin : first line n, then n lines "temp[K] den[g/cm3] ye"
! stdout: n lines "sele  srad  [erg/g/K]"  (sele includes e+ e- pairs)
! The electron/positron and radiation entropies do not depend on the ion
! composition (only on T, den*ye and T, den), so abar=1, zbar=ye is used.
! helm_table.dat must be in the working directory.

      integer          n,i,j,k
      double precision t,d,y

      call read_helm_table
      read(*,*) n
      i = 0
      do while (i .lt. n)
       k = min(nrowmax, n - i)
       do j=1,k
        read(*,*) t,d,y
        temp_row(j) = t ; den_row(j) = d
        abar_row(j) = 1.0d0 ; zbar_row(j) = y
       enddo
       jlo_eos = 1 ; jhi_eos = k
       call helmeos
       do j=1,k
        write(*,'(1p2e24.15)') sele_row(j), srad_row(j)
       enddo
       i = i + k
      enddo
      end program run_helm
