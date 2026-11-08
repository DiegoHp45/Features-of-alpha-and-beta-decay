      program run_nse
      include 'implno.dek'
      include 'const.dek'
      include 'network.dek'

! Non-interactive, single-shot NSE driver for scripting/subprocess use.
!
! Unlike public_nse.f90 (interactive REPL) this takes temp, den, ye as
! command-line arguments, computes the NSE mass-fraction vector, reduces it
! to (Abar, Zbar), and prints everything in a simple, fixed-format,
! machine-parseable block to stdout so a caller (e.g. nsm_traj.py) does not
! have to parse the pretty-printed public_nse.f90 REPL output.
!
! usage:
!   ./run_nse.exe <temp[K]> <den[g/cm^3]> <ye>
!
! stdout contract:
!   # header line
!   one line per isotope: name  aion  zion  xmass
!   ABAR  <value>
!   ZBAR  <value>
!   YE_CHECK  <value>          (should equal input ye to solver tolerance)
!
! exit codes: 0 = ok, 1 = bad usage, 2 = temp below NSE validity floor.
! nse() itself issues a bare Fortran STOP (whose exit code is compiler-
! dependent, commonly 0) if the root find fails to converge -- a caller
! should treat "process exited but printed no ABAR line" as failure too.

! declare
      integer          i,newguess,iprint,nargs
      character(len=32) arg
      double precision xtemp,xden,xye,abar,zbar

! iprint is kept at 0 below: nse()'s own diagnostics write to stdout too
! and would break the fixed-format, parseable block this program prints.
! nse() itself calls Fortran STOP if the root find fails to converge, so
! a nonzero process exit already signals that failure mode to a caller.

      nargs = command_argument_count()
      if (nargs .lt. 3) then
       write(0,'(a)') 'usage: run_nse.exe <temp[K]> <den[g/cm^3]> <ye>'
       stop 1
      end if

      call get_command_argument(1,arg); read(arg,*) xtemp
      call get_command_argument(2,arg); read(arg,*) xden
      call get_command_argument(3,arg); read(arg,*) xye

      call init_network

      if (xtemp .le. 2.0d9) then
       write(0,'(a)') 'error: temp <= 2.0e9 K, no NSE state'
       stop 2
      end if

! always use a fresh initial guess for a single-shot call
      newguess = 1
      iprint   = 0
      call nse(xtemp,xden,xye,newguess,xmass_nse,iprint)

! Abar (mean nucleon number) and Zbar (mean charge) of the NSE mixture,
! same convention used by the Helmholtz EOS driver (see
! ../HelmholtzEOS/helmholtz.f90):
!   1/Abar = sum_i X_i / A_i        Zbar = Abar * sum_i X_i Z_i / A_i
      abar = 1.0d0 / sum(xmass_nse(1:ionmax) / aion(1:ionmax))
      zbar = abar * sum(xmass_nse(1:ionmax) * zion(1:ionmax) / aion(1:ionmax))

      write(*,'(a)') '# name  aion  zion  xmass'
      do i=1,ionmax
       write(*,'(a5,2x,f6.1,2x,f6.1,2x,1pe16.8)') &
             ionam(i),aion(i),zion(i),xmass_nse(i)
      enddo

      write(*,'(a,1x,1pe16.8)') 'ABAR',abar
      write(*,'(a,1x,1pe16.8)') 'ZBAR',zbar
      write(*,'(a,1x,1pe16.8)') 'YE_CHECK',zbar/abar

      end program run_nse
