! FOR PYTHON, COMPILE WITH f2py --fcompiler=gnu95 --f90flags=-fopenmp --opt=-O2 -c -m readin readin.f90 -llapack -lblas -lgomp

!module pes_module
  ! this module should contain all data which are used to read in
  ! coordinates, gradients and Hessians and to write out the neural
  ! network data without being required by the neural network
  ! optimization itself
!  implicit none
!  integer :: nat
!  logical :: tinit=.false.
!  logical :: massweight
!!  real(8),allocatable :: align_refcoords(:) ! (3*nat)
!!  real(8),allocatable :: align_modes(:,:)   ! (3*nat,ncoord)
!!  real(8),allocatable :: align_modes_all(:,:,:)! (3*nat,ncoord,npoint+ntest)
!  real(8),allocatable :: refmass(:) ! (nat)
!  real(8),allocatable :: DMAT(:,:,:),DMAT_PINV(:,:,:),DMAT_PINV2(:,:,:),projection(:,:,:)
!  real(8),allocatable :: pca_eigvectr(:,:),xgradient_store(:,:),&
!  xhessian_store(:,:,:),xfreqs_store(:,:),xvectors_store(:,:,:),mu(:),variance(:)
!  real(8),allocatable :: barycentre(:,:)
!  real(8) pinv_tol,pinv_tol_back,minfreq,kmat_fac,evw
!  logical coords_interatomic
!  logical coords_inverse
!  logical coords_b_a_d
!  logical dimension_reduce
!  integer remove_n_dims,tol_type,extra_red_coords
!  integer dim_renorm
!  integer, allocatable:: radii_omit(:)
!  integer, allocatable:: bad_buildlist(:),bad_modes(:),atom_index(:)
!  real(8), allocatable:: bad_coords(:,:)
!  real(8),allocatable :: i_xs(:,:),i_gs(:,:),i_hs(:,:,:)
!  real(8),allocatable :: i_xtest(:,:),i_gtest(:,:),i_htest(:,:,:)
!  real(8),allocatable :: I_DMAT(:,:,:),I_DMAT_PINV(:,:,:)
!  real(8),allocatable :: KMAT(:,:,:),DM2(:,:,:,:),KMAT_Q(:,:,:)
!  real(8),allocatable :: i_pca_eigvectr(:,:),i_mu(:)
!!  real(8),allocatable :: i_align_refcoords(:) ! (3*nat)
!!  real(8),allocatable :: i_align_modes(:,:)   ! (3*nat,ncoord)
!!  real(8),allocatable :: i_align_modes_all(:,:,:)! (3*nat,ncoord,npoint+ntest)
!end module pes_module

subroutine pes_init(path,pathtest,ncoord_,npoint,npointtest,nat,dim_renorm,coords_interatomic,coords_inverse,&
  dimension_reduce,remove_n_dims,extra_red_coords)
  implicit none
  logical, intent(in)      :: coords_interatomic,coords_inverse,dimension_reduce
  character(*), intent(in) :: path     ! path to training set
  character(*), intent(in) :: pathtest ! path to test set
  integer, intent(out)     :: ncoord_,npoint,npointtest,nat,dim_renorm
  integer, intent(inout)   :: remove_n_dims
  integer, intent(in)      :: extra_red_coords
  character(256) :: line,fname
  integer :: ios
  integer :: ncoord,npointh

  write(*,'(a,a)') "# Initializing neural network interpolation from directory ",&
      trim(adjustl(path))

  ! create file list
  line="\ls -1 "//trim(adjustl(path))//"hess_*.txt > .tmp"
  call system(line)
  line="\ls -1 "//trim(adjustl(pathtest))//"hess_*.txt > .tmp_test"
  call system(line)
  !line="\ls -1 "//trim(adjustl(path))//"grad_*.txt > .gtmp"
  !call system(line)
  npoint=0
  open(unit=20,file=".tmp")
  ! read header to allocate arrays
  read(20,fmt='(a)',end=1000,err=1000) fname

  ! get number of atoms from first file
  open(unit=30,file=fname)
  read(30,fmt='(a)',end=1100,err=1100) line
  read(30,fmt='(a)',end=1100,err=1100) line
  read(line,*) nat
  print*,"# Number of atoms ",nat
  npoint=1
  ncoord=3*nat
  close(30)

!  if(.not.(coords_interatomic .or. coords_inverse))then
!    dim_renorm=ncoord_
!  else
    if(nat==1) then
      dim_renorm=3*nat
    else if (nat==2) then
      dim_renorm=3*nat-5
    else
      dim_renorm=3*nat-6
    end if
!  endif

  if(nat==1) then
    ncoord_=ncoord
  else if (nat==2) then
    ncoord_=ncoord-5 ! =1
  else
    ncoord_=ncoord-6 ! =1
  end if
  if(coords_interatomic .or. coords_inverse)ncoord_=ncoord_+extra_red_coords!nat*(nat-1)/2
  if(dimension_reduce)then
    ncoord_=ncoord_-remove_n_dims
  else
    remove_n_dims=0
  endif

  ! find out the number of files/points with Hessian information
  do 
    read(20,fmt='(a)',iostat=ios) fname
    if(ios/=0) exit
    npoint=npoint+1
  end do
  npointh=npoint
  print*,"# Number of files with Hessian information ",npointh
  close(20)

  open(unit=20,file=".tmp_test")
  npointtest=0
  do 
    read(20,fmt='(a)',iostat=ios) fname
    if(ios/=0) exit
    npointtest=npointtest+1
  end do
  print*,"# Number of files in test set              ",npointtest
  close(20)

!!$  ! find out the number of files/points with gradient information only
!!$  open(unit=21,file=".gtmp")
!!$  ! read header to allocate arrays
!!$  do 
!!$    read(21,fmt='(a)',iostat=ios) fname
!!$    if(ios/=0) exit
!!$    npoint=npoint+1
!!$  end do
!!$  print*,"# Number of files with gradient information ",npoint-npointh

  goto 1001
1000 print*,"Error reading from the file list"
  stop

1100 print*,"Error reading from file ",trim(adjustl(fname))
  stop
1001 continue

end subroutine pes_init

subroutine pes_read(ncoord_,npoint,ntest,nat,remove_n_dims,dim_renorm,coords_interatomic,coords_inverse,dimension_reduce,&
  refcoords,refene,refgrad,refhess,testcoords,testene,testgrad,testhess,xcoords_store,xgradient_store,xhessian_store,&
  xfreqs_store,xvectors_store,refmass,align_modes_all,align_refcoords,align_modes,DMAT,DMAT_PINV,DMAT_PINV2,projection,KMAT,&
  KMAT_Q,DM2,pca_eigvectr,mu,variance,radii_omit,pinv_tol,pinv_tol_back,tol_type)
  implicit none
  integer, intent(inout) :: ncoord_
  integer, intent(in)    :: nat,npoint,remove_n_dims,dim_renorm,tol_type
  integer, intent(inout) :: ntest
  logical, intent(in)    :: coords_interatomic,coords_inverse,dimension_reduce
  real(8),intent(out)    :: refcoords(ncoord_,npoint),refene(npoint),refgrad(ncoord_,npoint),refhess(ncoord_,ncoord_,npoint)
  real(8),intent(out)    :: testcoords(ncoord_,ntest),testene(ntest),testgrad(ncoord_,ntest),testhess(ncoord_,ncoord_,ntest)
  real(8),intent(out)    :: xcoords_store(3*nat,npoint+ntest),xgradient_store(3*nat,npoint+ntest),&
  xhessian_store(3*nat,3*nat,npoint+ntest),xfreqs_store(3*nat,npoint+ntest),xvectors_store(3*nat,3*nat,npoint+ntest),&
  align_modes_all(3*nat,dim_renorm,npoint+ntest),refmass(nat),align_refcoords(3*nat),align_modes(3*nat,dim_renorm),&
  DMAT(ncoord_+remove_n_dims,3*nat,npoint+ntest),DMAT_PINV(3*nat,ncoord_+remove_n_dims,npoint+ntest),&
  DMAT_PINV2(ncoord_+remove_n_dims,3*nat,npoint+ntest),projection(ncoord_+remove_n_dims,ncoord_+remove_n_dims,npoint+ntest),&
  KMAT(3*nat,3*nat,npoint+ntest),KMAT_Q(ncoord_+remove_n_dims,ncoord_+remove_n_dims,npoint+ntest),&
  DM2(3*nat,3*nat,ncoord_+remove_n_dims,npoint+ntest),&
  pca_eigvectr(ncoord_+remove_n_dims,ncoord_+remove_n_dims),mu(ncoord_+remove_n_dims),variance(ncoord_+remove_n_dims)
  integer, intent(out)   :: radii_omit(3*nat+1)
  real(8), intent(inout)    :: pinv_tol,pinv_tol_back
  real(8), allocatable   :: refcoords_PREPCA(:,:),refgrad_PREPCA(:,:),refhess_PREPCA(:,:,:)
  real(8), allocatable   :: testcoords_PREPCA(:,:),testgrad_PREPCA(:,:),testhess_PREPCA(:,:,:)
  real(8), allocatable   :: barycentre(:,:)
  character(256)         :: line,fname
  integer                :: ios,ipoint,fname_len ! point refers to control point
  real(8), allocatable   :: xcoords(:),xgradient(:),xhessian(:,:) ! coords directly read in from file
  real(8)                :: trans(3),rotmat(3,3),svar,ave_barycentre(3),ave_ene
  real(8), allocatable   :: distmat(:,:)!,STORE_DMAT(:,:,:)
  real(8), allocatable   :: modes_tmp(:,:)
  integer                :: jpoint,ivar,ncoord,npointh,arr2(2),ni_store,nzero
  logical                :: tok,superimpose=.true.
  real(8)                :: mass(nat),drotmat(3,3,3*nat),dcj_dxi(3*nat,3*nat),tmpvec(3)
  real(8)                :: mindist,bcentre_mwfact
  integer                :: ntest_diff,iat,jat,icoord
  real(8), allocatable   :: mxhessian(:,:)!,coord_store(:,:),evect(:,:),GMAT(:,:),STORE1(:,:),STORE2(:,:)
  real(8),allocatable    :: z_pca(:,:),z_pca_store(:,:),eigenvalues(:),eigenvector(:,:),&
  gs_store(:,:),hs_store(:,:,:),x_all(:,:),xs_store(:,:),xtest_store(:,:),avepos_allcoords(:)
  integer major_comp_keep,i
  logical exists

  ncoord=3*nat
  npointh=npoint
  nzero=ncoord-dim_renorm
  ! allocate local arrays
  allocate(xcoords(ncoord))
  allocate(xgradient(ncoord))
  allocate(xhessian(ncoord,ncoord))

  ! now read files
  open(unit=20,file=".tmp")
!  rewind(21)
  open(unit=52,file="training.xyz")
  open(unit=53,file="validation.xyz")
  allocate(barycentre(3,npoint+ntest))
  barycentre=0.d0
  ave_ene=0.d0
  do ipoint=1,npoint+ntest
    if(ipoint==npoint+1) then
      close(20)
      open(unit=20,file=".tmp_test")
    end if
!    if(ipoint<=npointh) then
    read(20,fmt='(a)',iostat=ios) fname
!    else
!      read(21,fmt='(a)',iostat=ios) fname
!    end if
    if(ios/=0) then
      print*,"Error getting file name"
      stop
    end if
    if(ipoint<=npoint) then
      write(*,"('Reference point',i3,' is ',a)") ipoint,trim(fname)
    else
      write(*,"('Test point',i3,' is ',a)") ipoint,trim(fname)
    end if
    open(unit=30,file=fname)

    read(30,fmt='(a)',end=1100,err=1100) line
    read(30,fmt='(a)',end=1100,err=1100) line
    read(30,fmt='(a)',end=1100,err=1100) line
    read(30,fmt='(a)',end=1100,err=1100) line

    if(ipoint<=npoint) then
      read(30,*) refene(ipoint)
      ave_ene=ave_ene+refene(ipoint)
    else
      read(30,*) testene(ipoint-npoint)
    end if
      
    read(30,fmt='(a)',end=1100,err=1100) line
    !print*,trim(adjustl(line))
    read(30,*) xcoords !refcoords(:,ipoint)

    read(30,fmt='(a)',end=1100,err=1100) line
    !print*,trim(adjustl(line))
    read(30,*) xgradient !refgrad(:,ipoint)

    read(30,fmt='(a)',end=1100,err=1100) line
    !print*,trim(adjustl(line))
    xhessian=0.D0
    read(30,*) xhessian 
    xcoords_store(:,ipoint)=xcoords
    xgradient_store(:,ipoint)=xgradient
    xhessian_store(:,:,ipoint)=xhessian

    if(ipoint.eq.1)then
      read(30,fmt='(a)',end=1100,err=1100) line
      refmass=0.D0
      read(30,*) refmass
      ! transform from a.u. to amu (does not matter, but leads to numerically more similar values to E and G)
      refmass=refmass/(1.66054D-27/9.10939D-31)
    endif

!    write(sample_no,'(I3.3)')ipoint
    fname_len=len(trim(fname))
!    i=fname_len
    i=4
    do while(fname(i-3:i).ne.'hess')
      i=i+1
    enddo
    inquire(file='input_configs/coords'//fname(i+1:fname_len-4)//'.xyz', exist = exists)
    if(exists) then
      open(unit=6500,file='input_configs/coords'//fname(i+1:fname_len-4)//'.xyz',status='old', action='write')
      call write_xyz(6500,nat,(nint(refmass)+1)/2,xcoords)
      close(6500)
    else
      open(unit=6500,file='input_configs/coords'//fname(i+1:fname_len-4)//'.xyz',status='new', action='write')
      call write_xyz(6500,nat,(nint(refmass)+1)/2,xcoords)
      close(6500)
    endif
    open(unit=6500,file='input_configs/E_G'//fname(i+1:fname_len-4)//'.xyz',status='replace')
    if(ipoint<=npoint) then
      write(6500,*) '#ENERGY'
      write(6500,*) refene(ipoint)
      write(6500,*) '#GRADIENT'
      do i=1,3*nat
        write(6500,*)xgradient(i)
      enddo
      write(6500,*) '#HESSIAN'
      do i=1,3*nat
        do iat=i,3*nat
          write(6500,*)xhessian(i,iat)
        enddo
      enddo
    else
      write(6500,*) '#ENERGY'
      write(6500,*) testene(ipoint-npoint)
      write(6500,*) '#GRADIENT'
      do i=1,3*nat
        write(6500,*)xgradient(i)
      enddo
      write(6500,*) '#HESSIAN'
      do i=1,3*nat
        do iat=i,3*nat
          write(6500,*)xhessian(i,iat)
        enddo
      enddo
    end if
    close(6500)

    bcentre_mwfact=1.d0
!    if(massweight)bcentre_mwfact=0.5d0
    do iat=1,nat
      barycentre(:,ipoint)=barycentre(:,ipoint)+refmass(iat)**bcentre_mwfact*xcoords(iat*3-2:iat*3)
    enddo
    barycentre(:,ipoint)=barycentre(:,ipoint)/sum(refmass,dim=1)
  enddo
  ave_ene=ave_ene/dble(npoint)
  refene=refene-ave_ene
  testene=testene-ave_ene
  allocate(avepos_allcoords(3*nat))
  avepos_allcoords=0.d0
  do ipoint=1,npoint+ntest
    avepos_allcoords=avepos_allcoords+xcoords_store(:,ipoint)
  enddo
  avepos_allcoords=avepos_allcoords/dble(npoint+ntest)

  allocate(refcoords_PREPCA(ncoord_+remove_n_dims,npoint))
  allocate(refgrad_PREPCA(ncoord_+remove_n_dims,npoint))
  allocate(refhess_PREPCA(ncoord_+remove_n_dims,ncoord_+remove_n_dims,npoint))
  allocate(testcoords_PREPCA(ncoord_+remove_n_dims,ntest))
  allocate(testgrad_PREPCA(ncoord_+remove_n_dims,ntest))
  allocate(testhess_PREPCA(ncoord_+remove_n_dims,ncoord_+remove_n_dims,ntest))
! now superimpose structure to first frame and transform derivatives

  do ipoint=1,npoint+ntest
    ! scoords=superimposed to reference structure
    xcoords=xcoords_store(:,ipoint)
    xgradient=xgradient_store(:,ipoint)
    xhessian=xhessian_store(:,:,ipoint)
    if(superimpose) then
      if(ipoint==1) then
        ! allocate array for module
        allocate(modes_tmp(3*nat,3*nat))
        mass=1.D0 ! no mass-weighting
!        if(massweight) then
!          mass=refmass
!        end if
        align_refcoords=xcoords
        ! mass-weight the Hessian
        allocate(mxhessian(ncoord,ncoord))
        do iat=1,nat
          do jat=1,nat
            mxhessian(iat*3-2:iat*3,jat*3-2:jat*3)=xhessian(iat*3-2:iat*3,jat*3-2:jat*3)/sqrt(mass(iat)*mass(jat))
          end do
        end do
        call dlf_thermal_project_readin(nat,mass,align_refcoords,mxhessian,ivar,modes_tmp,tok)
        deallocate(mxhessian)
        if(.not.tok) stop "Error in dlf_thermal_project_readin"
        print*,"# Number of spatial variables ",ivar
        if(ivar/=dim_renorm) then!if(ivar/=ncoord_) then
          print*,"Error in number of coordinates:"
          print*,"ncoord_=",dim_renorm!ncoord_
          print*,"number of coords required by projection=",ivar
          stop "error"
        end if
        align_modes=modes_tmp(:,1:dim_renorm)
        align_modes_all(:,:,ipoint)=modes_tmp(:,1:dim_renorm)
        deallocate(modes_tmp)
        ! now we have the modes. The coords are going to be transformed to mode elongations in cgh_xtos.

        !check if align_modes is orthogonal:
        !write(*,'("orthogonal?",6f10.5)') matmul(transpose(align_modes),align_modes)

      end if ! ipoint==1

      call cgh_xtos(nat,align_refcoords,xcoords,xgradient,xhessian,trans,rotmat)
      call get_drotmat(3*nat-6,dim_renorm,3*nat,avepos_allcoords,xcoords,align_modes,align_refcoords,drotmat)
      dcj_dxi=0.D0
      do iat=1,nat
        dcj_dxi(iat*3-2:iat*3,iat*3-2:iat*3)=transpose(rotmat)
      end do

      do iat=1,nat
        do jat=1,3*nat
          tmpvec=matmul(drotmat(:,:,jat),(xcoords(iat*3-2:iat*3)-barycentre(:,ipoint)))
          dcj_dxi(jat,iat*3-2:iat*3)= dcj_dxi(jat,iat*3-2:iat*3) + tmpvec
        end do
      end do
!        xhessian_store(:,:,ipoint)=matmul(matmul(dcj_dxi,xhessian),transpose(dcj_dxi))

    end if

    close(30)

!    refcoords(:,ipoint)=xcoords
!    refgrad(:,ipoint)=xgradient
!    if(ipoint<=npointh) refhess(:,:,ipoint)=xhessian

  end do
  if(coords_interatomic .or. coords_inverse)then
    align_refcoords=avepos_allcoords
  endif
  deallocate(avepos_allcoords)
  ave_barycentre=0.d0
  do ipoint=1,npoint+ntest
    ave_barycentre=ave_barycentre+barycentre(:,ipoint)
  enddo
  deallocate(barycentre)
  ave_barycentre=ave_barycentre/dble(npoint+ntest)
!  do iat=1,nat
!    do ipoint=1,npoint+ntest
!      xcoords_store(3*iat-2:3*iat,ipoint)=xcoords_store(3*iat-2:3*iat,ipoint)-&
!      ave_barycentre
!    enddo
!  enddo
  do iat=1,nat
    xcoords_store(iat*3-2:iat*3,:)=xcoords_store(iat*3-2:iat*3,:)/sqrt(mass(iat))
    xgradient_store(iat*3-2:iat*3,:)=xgradient_store(iat*3-2:iat*3,:)/sqrt(mass(iat))
    do jat=1,nat
      xhessian_store(iat*3-2:iat*3,jat*3-2:jat*3,:)=&
      xhessian_store(iat*3-2:iat*3,jat*3-2:jat*3,:)/sqrt(mass(iat)*mass(jat))
    enddo
  enddo
  allocate(eigenvalues(3*nat))
  allocate(eigenvector(3*nat,3*nat))
  do ipoint=1,npoint+ntest
    call r_diagonal(3*nat,xhessian_store(:,:,ipoint),eigenvalues,eigenvector)
    xfreqs_store(:,ipoint)=eigenvalues
    xvectors_store(:,:,ipoint)=eigenvector
  enddo
  deallocate(eigenvalues)
  deallocate(eigenvector)
  if(coords_interatomic .or. coords_inverse)then
    call interatomic_coord_conversion(nat,ncoord_+remove_n_dims,npoint,ntest,xcoords_store,xgradient_store,xhessian_store,&
    refcoords_PREPCA,refgrad_PREPCA,refhess_PREPCA,testcoords_PREPCA,testgrad_PREPCA,testhess_PREPCA,&
    DMAT,DMAT_PINV,DMAT_PINV2,projection,KMAT,KMAT_Q,DM2,radii_omit,&
    pinv_tol,pinv_tol_back,coords_inverse,dim_renorm,xfreqs_store,tol_type,refmass)
  endif

  ni_store=ncoord_+remove_n_dims
  if(dimension_reduce)then
    allocate(x_all(ni_store,npoint+ntest))
    x_all(:,1:npoint)=refcoords_PREPCA
    x_all(:,npoint+1:ntest+npoint)=testcoords_PREPCA
    allocate(z_pca(ni_store,npoint+ntest))!THESE ARE THE REDUCED COORDS
    pca_eigvectr=0.d0
    call input_preprocessing(ni_store,npoint,ntest,x_all,z_pca,major_comp_keep,pca_eigvectr,&
    remove_n_dims,mu,variance)
    deallocate(x_all)
    allocate(z_pca_store(ncoord_,npoint+ntest))
    z_pca_store=z_pca(1:ncoord_,:)
    deallocate(z_pca)
    allocate(z_pca(ncoord_,npoint))
    z_pca=z_pca_store(:,1:npoint)
    allocate(xs_store(ncoord_,npoint))
    allocate(xtest_store(ncoord_,ntest))
    xs_store=refcoords_PREPCA
    xtest_store=testcoords_PREPCA
    deallocate(refcoords_PREPCA)
    deallocate(testcoords_PREPCA)
    allocate(refcoords_PREPCA(ncoord_,npoint))
    allocate(testcoords_PREPCA(ncoord_,ntest))
    refcoords_PREPCA=z_pca
    deallocate(z_pca)
    allocate(z_pca(ncoord_,ntest))
    z_pca=z_pca_store(:,1+npoint:npoint+ntest)
    testcoords_PREPCA=z_pca
    deallocate(z_pca)
    allocate(gs_store(ni_store,npoint+ntest))
    gs_store=0.d0
    allocate(hs_store(ni_store,ni_store,npoint+ntest))
    hs_store=0.d0  
    do i=1,npoint
      gs_store(:,i)=refgrad_PREPCA(:,i)
      hs_store(:,:,i)=refhess_PREPCA(:,:,i)
    enddo
    do i=npoint+1,npoint+ntest
      gs_store(:,i)=testgrad_PREPCA(:,i-npoint)
      hs_store(:,:,i)=testhess_PREPCA(:,:,i-npoint)
    enddo    
    deallocate(refgrad_PREPCA)
    deallocate(refhess_PREPCA)
    allocate(refgrad_PREPCA(ncoord_,npoint))
    allocate(refhess_PREPCA(ncoord_,ncoord_,npoint))
    do i=1,npoint
      refgrad_PREPCA(:,i)=matmul(transpose(pca_eigvectr(:,1:ncoord_)),gs_store(:,i))
      refhess_PREPCA(:,:,i)=matmul(matmul(transpose(pca_eigvectr(:,1:ncoord_)),hs_store(:,:,i)),pca_eigvectr(:,1:ncoord_))
    enddo
    deallocate(testgrad_PREPCA)
    deallocate(testhess_PREPCA)
    allocate(testgrad_PREPCA(ncoord_,ntest))
    allocate(testhess_PREPCA(ncoord_,ncoord_,ntest))
    do i=npoint+1,npoint+ntest
      testgrad_PREPCA(:,i-npoint)=matmul(transpose(pca_eigvectr(:,1:ncoord_)),gs_store(:,i))
      testhess_PREPCA(:,:,i-npoint)=matmul(matmul(transpose(pca_eigvectr(:,1:ncoord_)),hs_store(:,:,i)),pca_eigvectr(:,1:ncoord_))
    enddo
  endif
  refcoords(1:ncoord_,:)=refcoords_PREPCA
  refgrad(1:ncoord_,:)=refgrad_PREPCA
  refhess(1:ncoord_,1:ncoord_,:)=refhess_PREPCA
  testcoords(1:ncoord_,:)=testcoords_PREPCA
  testgrad(1:ncoord_,:)=testgrad_PREPCA
  testhess(1:ncoord_,1:ncoord_,:)=testhess_PREPCA

  deallocate(refcoords_PREPCA)
  deallocate(refgrad_PREPCA)
  deallocate(refhess_PREPCA)
  deallocate(testcoords_PREPCA)
  deallocate(testgrad_PREPCA)
  deallocate(testhess_PREPCA)

  close(20) ! .tmp
  close(52)
  close(53)
!  close(21) ! .gtmp

  call system("rm -f .tmp .gtmp .tmp_test")

  goto 1001
1100 print*,"Error reading from file ",trim(adjustl(fname))
  stop
1001 continue

  deallocate(xcoords)
  deallocate(xgradient)
  deallocate(xhessian)
  
  ! now get information about mutal distances of control points. Maybe one can merge some?
  if(npoint>1) then
    allocate(distmat(npoint,npoint))
    distmat=0.D0
    do ipoint=1,npoint
      do jpoint=ipoint+1,npoint
        distmat(ipoint,jpoint)=sum((refcoords(:,ipoint)-refcoords(:,jpoint))**2)
      end do
    end do
    ! set lower half and diagonal to averag value of distmat, so that it
    ! does not disturb the rest
    svar=sum(distmat)/(npoint*(npoint-1)/2)
    do ipoint=1,npoint
      do jpoint=1,ipoint
        distmat(ipoint,jpoint)=svar
      end do
    end do
    write(*,'(" # Minimum distance between two points: ",es12.4,", control points",2i6)') minval(distmat),minloc(distmat)
    write(*,'(" # Maximum distance between two points: ",es12.4,", control points",2i6)') maxval(distmat),maxloc(distmat)
    print*,"List of shortest distances:"
    print*,"Number       Distance              Control Points"
    do ipoint=1,min(10,npoint)
      print*,ipoint, minval(distmat),minloc(distmat)
      arr2=minloc(distmat)
      distmat(arr2(1),arr2(2))=huge(1.D0)
    end do
    
    deallocate(distmat)
    allocate(distmat(ntest,ntest))
    distmat=0.D0
    do ipoint=1,ntest
      do jpoint=ipoint+1,ntest
        distmat(ipoint,jpoint)=sum((testcoords(:,ipoint)-testcoords(:,jpoint))**2)
      end do
    end do
    ! set lower half and diagonal to averag value of distmat, so that it
    ! does not disturb the rest
    svar=sum(distmat)/(ntest*(ntest-1)/2)
    do ipoint=1,ntest
      do jpoint=1,ipoint
        distmat(ipoint,jpoint)=svar
      end do
    end do
    write(*,'(" # Minimum distance between two points: ",es12.4,", control points",2i6)') minval(distmat),minloc(distmat)
    write(*,'(" # Maximum distance between two points: ",es12.4,", control points",2i6)') maxval(distmat),maxloc(distmat)
    print*,"List of shortest distances:"
    print*,"Number       Distance              Control Points"
    do ipoint=1,min(10,ntest)
      print*,ipoint, minval(distmat),minloc(distmat)
      arr2=minloc(distmat)
      distmat(arr2(1),arr2(2))=huge(1.D0)
    end do
    
    deallocate(distmat)
  end if

  !print*,"# to be done: make sure that test and trainig set are mutually exclusive"
  ntest_diff=0
  do ipoint=npoint+1,npoint+ntest ! test set
    mindist=huge(1.D0)
    !ntest_diff=0
    do jpoint=1,npoint ! training set
      svar=sum((testcoords(:,ipoint-npoint)-refcoords(:,jpoint))**2)
      if(svar<mindist) then
        mindist=svar
        !ntest_diff=jpoint
      end if
    end do
    !print*,"point",ipoint,"mindist",mindist,ntest_diff   
    if(mindist>-1.D-6) then ! negative value: test deactivated!
      ntest_diff=ntest_diff+1
      !print*,"copying",ipoint," to ",ntest_diff
      testcoords(:,ntest_diff)=testcoords(:,ipoint-npoint)
      testene(ntest_diff)=testene(ipoint-npoint)
      testgrad(:,ntest_diff)=testgrad(:,ipoint-npoint)
      testhess(:,:,ntest_diff)=testhess(:,:,ipoint-npoint)
    else
      print*,"Test point",ipoint," excluded because it is too close to a training point"
    end if
    !print*,"point",ipoint,"mindist",mindist,ntest_diff
  end do
  ntest=ntest_diff
  print*,"# Initialisation of neural network interpolation finished"
  !tinit=.true.

  ! print the internal coordinates (refcoords, testcoords) as parallel coordintes
  open(unit=52,file="trainingset.pc")
  write(52,'(a)') "# internal coordinates of training set as parallel coordinates, one geometry after the next"
  do ipoint=1,npoint
    write(52,'(a,i4)') "# Reference point ",ipoint
    do icoord=1,ncoord_
      write(52,'(i4,f15.10)') icoord,refcoords(icoord,ipoint)
    end do
    write(52,*) 
  end do
  close(52)

  ! print the internal coordinates (refcoords, testcoords) as parallel coordintes
  open(unit=52,file="testset.pc")
  write(52,'(a)') "# internal coordinates of test/validation set as parallel coordinates, one geometry after the next"
  do ipoint=npoint+1,npoint+ntest ! test set
    write(52,'(a,i4)') "# Test point ",ipoint
    do icoord=1,ncoord_
      write(52,'(i4,f15.10)') icoord,testcoords(icoord,ipoint-npoint)
    end do
    write(52,*) 
  end do
  close(52)

  call print_coords(ncoord_,3*nat,npoint,refcoords,xcoords_store,refene)

end subroutine pes_read

subroutine write_xyz(unit,nat,znuc,coords)
  implicit none
  integer,intent(in) :: unit
  integer,intent(in) :: nat
  integer,intent(in) :: znuc(nat)
  real(8),intent(in):: coords(3,nat)
  integer            :: iat
  character(2)       :: str2
  real(8)           :: ang_au
  character(2), external :: get_atom_symbol
! **********************************************************************

  ang_au=5.2917720810086d-01
  write(unit,*) nat
  write(unit,*)
  do iat=1,nat
    str2 = get_atom_symbol(znuc(iat))
    write(unit,'(a2,3f12.7)') str2,coords(:,iat)*ang_au
 ! temporary: commented out cartesian conversion
 !   write(unit,'(a2,3f12.7)') str2,coords(:,iat)
  end do
  call flush(unit)
end subroutine write_xyz

subroutine dlf_thermal_project_readin(nat,mass,coords,hessian,npmodes,pmodes,tok)
!! SOURCE
!  use dlf_parameter_module, only: rk
!  use dlf_global, only: glob,stdout,printl
  implicit none
  real(8), external :: ddot
  integer, intent(in) :: nat
  real(8),intent(in)  :: mass(nat)
  real(8),intent(in)  :: coords(3*nat)
  real(8),intent(in)  :: hessian(3*nat,3*nat)
  integer, intent(out)  :: npmodes ! number of vibrational modes
  real(8) :: peigval(3*nat) ! eigenvalues after projection
  real(8), intent(out)  :: pmodes(3*nat, 3*nat) ! vib modes after proj. (non-mass-weighted) 
  logical               :: tok
  real(8)              :: comcoords(3,nat) ! centre of mass coordinates
  real(8)              :: com(3) ! centre of mass
  real(8)              :: totmass ! total mass
  real(8)              :: moi(3,3) ! moment of inertia tensor
  real(8)              :: moivec(3,3) ! MOI eigenvectors
  real(8)              :: moival(3) ! MOI eigenvalues
  real(8)              :: transmat(3*nat,3*nat) ! transformation matrix
  real(8)              :: px(3), py(3), pz(3)
  real(8)              :: smass
  real(8), parameter   :: mcutoff = 1.0d-12
  integer               :: ntrro ! number of trans/rot modes
  real(8)              :: test, norm
  real(8)              :: trialv(3*nat)
  real(8)              :: phess(3*nat,3*nat) ! projected Hessian
  real(8)              :: peigvec(3*nat, 3*nat) ! eigenvectors after proj.
!  real(8)              :: pmodes(3*nat, 3*nat) ! vib modes after proj.
  integer               :: pstart
  integer               :: ival, jval, kval, lval, icount
  integer              :: printl=4
! **********************************************************************
  tok=.false.
  ! Do not continue if any coordinates are frozen
  if (nat==1) then
     write(*,*)
     write(*,"('Frozen atoms found: no modes will be projected out')")
     npmodes = 3*nat
     !peigval = eigval
     return
  end if

  write(*,*)
  write(*,"('Projecting out translational and rotational modes')")

  ! Calculate centre of mass and moment of inertia tensor

!  ! xcoords is not fully up to date so convert icoords instead
!  call dlf_cartesian_itox(nat, 3*nat, glob%nicore, &
!       glob%massweight, glob%icoords, comcoords)
  comcoords=reshape(coords,(/3,nat/))
  com(:) = 0.0d0
  totmass = 0.0d0
  do ival = 1, nat
     com(1:3) = com(1:3) + mass(ival) * comcoords(1:3, ival)
     totmass = totmass + mass(ival)
  end do
  com(1:3) = com(1:3) / totmass

  do ival = 1, nat
     comcoords(1:3, ival) = comcoords(1:3, ival) - com(1:3)
  end do

  moi(:,:) = 0.0d0
  do ival = 1, nat
     moi(1,1) = moi(1,1) + mass(ival) * &
          (comcoords(2,ival) * comcoords(2,ival) + comcoords(3,ival) * comcoords(3,ival))
     moi(2,2) = moi(2,2) + mass(ival) * &
          (comcoords(1,ival) * comcoords(1,ival) + comcoords(3,ival) * comcoords(3,ival))
     moi(3,3) = moi(3,3) + mass(ival) * &
          (comcoords(1,ival) * comcoords(1,ival) + comcoords(2,ival) * comcoords(2,ival))
     moi(1,2) = moi(1,2) - mass(ival) * comcoords(1, ival) * comcoords(2, ival)
     moi(1,3) = moi(1,3) - mass(ival) * comcoords(1, ival) * comcoords(3, ival)
     moi(2,3) = moi(2,3) - mass(ival) * comcoords(2, ival) * comcoords(3, ival)
  end do
  moi(2,1) = moi(1,2)
  moi(3,1) = moi(1,3)
  moi(3,2) = moi(2,3)

  call matrix_diagonalise(3, moi, moival, moivec)

  if (printl >= 6) then
     write(*,"(/,'Centre of mass'/3f15.5)") com(1:3)
     write(*,"('Moment of inertia tensor')")
     write(*,"(3f15.5)") moi(1:3, 1:3)
     write(*,"('Principal moments of inertia')")
     write(*,"(3f15.5)") moival(1:3)
     write(*,"('Principal axes')")
     write(*,"(3f15.5)") moivec(1:3, 1:3)
  end if

  ! Construct transformation matrix to internal coordinates
  ntrro = 6
  transmat(:, :) = 0.0d0
  do ival = 1, nat
     smass = sqrt(mass(ival))
     kval = 3 * (ival - 1)
     ! Translational vectors
     transmat(kval+1, 1) = smass
     transmat(kval+2, 2) = smass
     transmat(kval+3, 3) = smass
     ! Rotational vectors
     px = sum(comcoords(1:3,ival) * moivec(1:3,1))
     py = sum(comcoords(1:3,ival) * moivec(1:3,2))
     pz = sum(comcoords(1:3,ival) * moivec(1:3,3))
     transmat(kval+1:kval+3, 4) = (py*moivec(1:3,3) - pz*moivec(1:3,2))*smass
     transmat(kval+1:kval+3, 5) = (pz*moivec(1:3,1) - px*moivec(1:3,3))*smass
     transmat(kval+1:kval+3, 6) = (px*moivec(1:3,2) - py*moivec(1:3,1))*smass
  end do
  ! Normalise vectors and check for linear molecules (one less mode)
  do ival = 1, 6
     test = sum(transmat(:,ival)**2)  !ddot(3*nat, transmat(1,ival), 1, transmat(1,ival), 1)
     if (test < mcutoff) then
        kval = ival
        ntrro = ntrro - 1
        if (ntrro < 5) then
           write(*,"('Error: too few rotational/translation modes')")
           npmodes = 3*nat
           !peigval = eigval
           return
        end if
     else
        norm = 1.0d0/sqrt(test)
        call dscal(3*nat, norm, transmat(1,ival), 1)
     end if
  end do
  if (ntrro == 5 .and. kval /= 6) then
     transmat(:, kval) = transmat(:, 6)
     transmat(:, 6) = 0.0d0
  end if
  write(*,"(/,'Number of translational/rotational modes:',i4)") ntrro

  ! Generate 3N-ntrro other orthogonal vectors 
  ! Following the method in OPTvibfrq
  icount = ntrro
  do ival = 1, 3*nat
     trialv(:) = 0.0d0
     trialv(ival) = 1.0d0
     do jval = 1, icount
        ! Test if trial vector is linearly independent of previous set
        test = -sum(transmat(:,jval)*trialv(:)) !-ddot(3*nat, transmat(1,jval), 1, trialv, 1)
        call daxpy(3*nat, test, transmat(1,jval), 1, trialv, 1)
     end do
     test = ddot(3*nat, trialv, 1, trialv, 1)
     if (test > mcutoff) then
        icount = icount + 1
        norm = 1.0d0/sqrt(test)
        transmat(1:3*nat, icount) = norm * trialv(1:3*nat)
     end if
     if (icount == 3*nat) exit
  end do
  if (icount /= 3*nat) then
     write(*,"('Error: unable to generate transformation matrix')")
     npmodes = 3*nat
     !peigval = eigval
     return
  end if
  if (printl >= 6) then
     write(*,"(/,'Transformation matrix')")
     !call dlf_matrix_print(3*nat, 3*nat, transmat)
  end if

  ! Apply transformation matrix: D(T) H D
  ! Use peigvec as scratch to store intermediate
  phess(:,:) = 0.0d0
  peigvec(:,:) = 0.0d0
  call matrix_multiply(3*nat, 3*nat, 3*nat, &
       1.0d0,hessian, transmat, 0.0d0, peigvec)
  ! Should alter matrix_multiply to allow transpose option to be set...
  transmat = transpose(transmat)
  call matrix_multiply(3*nat, 3*nat, 3*nat, &
       1.0d0, transmat, peigvec, 0.0d0, phess)
  transmat = transpose(transmat)

  if (printl >= 6) then
     write(*,"(/,'Hessian matrix after projection:')")
     !call dlf_matrix_print(3*nat, 3*nat, phess)
  end if

  ! Find eigenvalues of Nvib x Nvib submatrix
  peigval(:) = 0.0d0
  peigvec(:,:) = 0.0d0
  npmodes = 3*nat - ntrro
  pstart = ntrro + 1
  call matrix_diagonalise(npmodes, phess(pstart:3*nat, pstart:3*nat), &
       peigval(1:npmodes), peigvec(1:npmodes,1:npmodes))

  if (printl >= 6) then
     write(*,"('Vibrational submatrix eigenvalues:')")
     write(*,"(12f9.5)") peigval(1:npmodes)
!     write(*,"('Vibrational submatrix eigenvectors:')")
     !call dlf_matrix_print(npmodes, npmodes, peigvec(1:npmodes, 1:npmodes))
  end if

  ! Print out normalised normal modes
  ! These are in non-mass-weighted Cartesians (division by smass)
  pmodes(:,:) = 0.0d0
  do kval = 1, 3*nat
     do ival = 1, npmodes
        do jval = 1, npmodes
           pmodes(kval, ival) = pmodes(kval, ival) + &
                transmat(kval, ntrro + jval) * peigvec(jval, ival)
        end do
        lval = (kval - 1) / 3 + 1
        smass = sqrt(mass(lval))
        ! the next line must be commented out if pmodes should be returned in mass-weighted cartesians
        !if(.not.massweight) pmodes(kval, ival) = pmodes(kval, ival) / smass
     end do
  end do
  do ival = 1, npmodes
     test = ddot(3*nat, pmodes(1,ival), 1, pmodes(1,ival), 1)
     norm = 1.0d0 / sqrt(test)
     call dscal(3*nat, norm, pmodes(1,ival), 1)
  end do

  if (printl >= 4) then
     write(*,"(/,'Normalised normal modes (Cartesian coordinates):')")
     !call dlf_matrix_print(3*nat, npmodes, pmodes(1:3*nat, 1:npmodes))
  end if
  
  tok=.true.

end subroutine dlf_thermal_project_readin

SUBROUTINE matrix_diagonalise(N,H,E,U) 
  IMPLICIT NONE
  LOGICAL(4) ,PARAMETER :: TESSLERR=.FALSE.
  INTEGER   ,INTENT(IN) :: N
  REAL(8)   ,INTENT(IN) :: H(N,N)
  REAL(8)   ,INTENT(OUT):: E(N)
  REAL(8)   ,INTENT(OUT):: U(N,N)
  REAL(8)               :: WORK1((N*(N+1))/2)
  REAL(8)               :: WORK2(3*N)
  INTEGER               :: K,I,J
  !CHARACTER(8)          :: SAV2101
  INTEGER               :: INFO
  
  K=0
  DO J=1,N
    DO I=J,N
      K=K+1
      WORK1(K)=0.5D0*(H(I,J)+H(J,I))
    ENDDO
  ENDDO
  
  CALL dspev('V','L',N,WORK1,E,U,N,WORK2,INFO)
  IF(INFO.NE.0) THEN
    PRINT*,'DIAGONALIZATION NOT CONVERGED'
    STOP
  END IF
  
END SUBROUTINE MATRIX_DIAGONALISE

subroutine cgh_xtos(nat,rcoords,xcoords,xgradient,xhessian,trans,rotmat)
  implicit none
  integer,intent(in) :: nat
  real(8),intent(in) :: rcoords(3*nat) ! the set of coordinates the new ones should be fitted to
  real(8),intent(inout) :: xcoords(3*nat) ! could be made intent(in), it is easier that way
  real(8),intent(inout) :: xgradient(3*nat)
  real(8),intent(inout) :: xhessian(3*nat,3*nat)
  real(8),intent(out)   :: trans(3),rotmat(3,3)
  real(8), allocatable :: eigval_l(:)
  integer:: iat,ivar,jvar,jat
  real(8) :: rmat(3,3),rsmat(3,3),eigvec(3,3),eigval(3)
  real(8) :: center(3)
  real(8) :: weight(3*nat)

  integer :: itry,i,j
  real(8) :: detrot
  
!  if(massweight) then
!    do iat=1,nat
!      weight(iat*3-2:iat*3)=refmass(iat)
!    end do
!  else
    weight=1.D0
!  end if


  ! as compared to dlf_cartesian_align: coords1=rcoords coords2=coords

  trans=0.D0
  rotmat=0.D0
  do ivar=1,3
    rotmat(ivar,ivar)=1.D0
  end do

  ! if there are other cases to ommit a transformation, add them here
  if(nat==1) return
  !if(.not.superimpose) return

  trans=0.D0
  center=0.D0
  do iat=1,nat
    center(:)=center(:)+rcoords(iat*3-2:iat*3)*weight(iat*3-2:iat*3)
    trans(:)=trans(:)+(xcoords(iat*3-2:iat*3)-rcoords(iat*3-2:iat*3))*weight(iat*3-2:iat*3)
  end do
  trans=trans/sum(weight)*3.D0
  center=center/sum(weight)*3.D0

  !print*,"# trans",trans

  ! translate them to common centre
  do iat=1,nat
    xcoords(iat*3-2:iat*3)=xcoords(iat*3-2:iat*3)-trans(:)
  end do

  rmat=0.D0
  do iat=1,nat
    do ivar=1,3
      do jvar=1,3
        rmat(ivar,jvar)=rmat(ivar,jvar)+weight(3*iat)*(rcoords(ivar+3*iat-3)-center(ivar))* &
            (xcoords(jvar+3*iat-3)-center(jvar))
      end do
    end do
  end do
  rmat=rmat/sum(weight)*3.D0
  !write(*,"('R   ',3f10.3)") rmat
  rsmat=transpose(rmat)
  eigvec=matmul(rsmat,rmat)
  rsmat=eigvec

  !write(stdout,"('RtR ',3f10.3)") rsmat
  call matrix_diagonalise(3,rsmat,eigval,eigvec)
  allocate(eigval_l(3))
!  call r_diagonal_general(3,rsmat,eigval,eigval_l,eigvec)
  deallocate(eigval_l)

  ! It turns out that the rotation matrix may have a determinat of -1
  ! in the procedure used here, i.e. the system is mirrored - which is
  ! wrong chemically. This can be avoided by inserting a minus in the
  ! equation
  ! 1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))

  ! So, here we first calculate the rotation matrix, and if it is
  ! zero, the first eigenvalue is reversed

  do itry=1,2
    ! rsmat are the vectors b:
    j=-1
    do i=1,3
      if(eigval(i)<1.D-8) then
        if(i>1) then
          ! the system is linear - no rotation necessay.
          ! WHY ?! There should still be one necessary!
          return
          !print*,"Eigenval. zero",i,eigval(i)
          !call dlf_fail("Error in dlf_cartesian_align")
        end if
        j=1
      else
        if(i==1.and.itry==2) then
          rsmat(:,i)=-1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
        else
          rsmat(:,i)=1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
        end if
      end if
    end do
    if(j==1) then
      ! one eigenvalue was zero, the system is planar
      rsmat(1,1)=rsmat(2,2)*rsmat(3,3)-rsmat(3,2)*rsmat(2,3)
      rsmat(2,1)=rsmat(3,2)*rsmat(1,3)-rsmat(1,2)*rsmat(3,3)
      rsmat(3,1)=rsmat(1,2)*rsmat(2,3)-rsmat(2,2)*rsmat(1,3)
      ! deal with negative determinant
      if (itry==2) then
         rsmat(:,1) = -rsmat(:,1)
      end if
    end if

    do i=1,3
      do j=1,3
        rotmat(i,j)=sum(rsmat(i,:)*eigvec(j,:))
      end do
    end do
    !write(*,"('rotmat ',3f10.3)") rotmat
    detrot=   &
        rotmat(1,1)*(rotmat(2,2)*rotmat(3,3)-rotmat(2,3)*rotmat(3,2)) &
        -rotmat(2,1)*(rotmat(1,2)*rotmat(3,3)-rotmat(1,3)*rotmat(3,2)) &
        +rotmat(3,1)*(rotmat(1,2)*rotmat(2,3)-rotmat(1,3)*rotmat(2,2))
    !write(*,*) "Determinat of rotmat", detrot
    if(detrot > 0.D0) exit
    if(detrot < 0.D0 .and. itry==2) then
      stop "Error in dlf_cartesian_align, obtained a mirroring instead of rotation."
    end if

  end do


!!$  do ivar=1,3
!!$    rsmat(:,ivar)=1.d0/dsqrt(eigval(ivar)) * matmul(rmat,eigvec(:,ivar))
!!$  end do
!!$
!!$  do ivar=1,3
!!$    do jvar=1,3
!!$      rotmat(ivar,jvar)=sum(rsmat(ivar,:)*eigvec(jvar,:))
!!$    end do
!!$  end do

  ! transform coordinates
  do iat=1,nat
    xcoords(iat*3-2:iat*3)= xcoords(iat*3-2:iat*3)-center
    xcoords(iat*3-2:iat*3)=matmul(rotmat,xcoords(iat*3-2:iat*3))
    xcoords(iat*3-2:iat*3)= xcoords(iat*3-2:iat*3)+center
  end do

  ! write xyz
!  if(ttrain) then
!    fid=52
!  else
!    fid=53
!  end if
!  write(fid,*) nat
!  write(fid,*) 
!  do iat=1,nat
!    write(fid,'(" H ",3f12.7)') xcoords(iat*3-2:iat*3)*5.2917720810086D-01
!  end do
  
!  print*,"transformed coordinates"
!  write(*,'(3f15.5)') coords
  
  ! transform gradient
  do iat=1,nat
    xgradient(iat*3-2:iat*3)=matmul(rotmat,xgradient(iat*3-2:iat*3))
  end do
  !print*,"transformed gradient"
  !write(*,'(3f15.5)') gradient

  !print*,"rotation matrix"
  !write(*,'(3f15.5)') rotmat

  ! transform hessian
  do iat=1,nat
    do jat=1,nat
      xhessian(iat*3-2:iat*3,jat*3-2:jat*3)=matmul(matmul(rotmat,xhessian(iat*3-2:iat*3,jat*3-2:jat*3)),transpose(rotmat)) 
    end do
  end do

!!$  print*,"transformed hessian"
!!$  do ivar=1,6
!!$    write(*,'(3es13.5,2x,3es13.5,2x,3es13.5)') hessian(ivar,1:9)
!!$    if(ivar==3) print*
!!$  end do

  ! now all quantities have been transformed to c-coords (or relative to c-coordinates)

  ! now, the coordinates need to be mass-weighted!

!  if(massweight) then
!    do iat=1,nat
!      xgradient(iat*3-2:iat*3)=xgradient(iat*3-2:iat*3)/sqrt(refmass(iat))
!      do jat=1,nat
!        xhessian(iat*3-2:iat*3,jat*3-2:jat*3)=&
!            xhessian(iat*3-2:iat*3,jat*3-2:jat*3)/sqrt(refmass(iat))/sqrt(refmass(jat))
!      end do
!    end do
!  end if

end subroutine cgh_xtos

subroutine matrix_multiply(M,N,K,alpha,A,B,beta,C)
  !use dlf_parameter_module, only: rk
  implicit none
  integer  ,intent(in)    :: M,N,K
  real(8) ,intent(in)    :: alpha,beta
  real(8) ,intent(in)    :: A(M,K) ! K is the common index
  real(8) ,intent(in)    :: B(K,N)
  real(8) ,intent(inout) :: C(M,N)
!! SOURCE
! **********************************************************************
  CALL dgemm('N','N',M,N,K,alpha, A , M, B, K, beta, C, M)
end subroutine matrix_multiply

subroutine get_drotmat(ncoord,dim_renorm,nvar,rcoords,xcoords_,align_modes,align_refcoords,drotmat)
  implicit none
  integer,intent(in) :: ncoord,nvar,dim_renorm
  real(8),intent(in) :: rcoords(nvar) ! the set of coordinates the new ones should be fitted to
  real(8),intent(in) :: xcoords_(nvar),align_modes(nvar,dim_renorm),align_refcoords(nvar)
  real(8),intent(out) :: drotmat(3,3,nvar)
  real(8) :: trans(3),dcoords(ncoord),tmpmat(3,3)
  integer :: ivar
  real(8) :: delta=1.D-5
  real(8) :: xcoords(nvar)
  !print*,"FD rotmat with delta",delta
  do ivar=1,nvar
    xcoords=xcoords_
    xcoords(ivar)=xcoords(ivar)+delta
    call cgh_xtos_simple(ncoord,dim_renorm,nvar,rcoords,xcoords,trans,drotmat(:,:,ivar),align_modes,align_refcoords,dcoords)
    xcoords(ivar)=xcoords(ivar)-2.D0*delta
    call cgh_xtos_simple(ncoord,dim_renorm,nvar,rcoords,xcoords,trans,tmpmat,align_modes,align_refcoords,dcoords)
    drotmat(:,:,ivar)=(drotmat(:,:,ivar)-tmpmat)/2.D0/delta
  end do
end subroutine get_drotmat

subroutine cgh_xtos_simple(ncoord,dim_renorm,nvar,rcoords,xcoords_,trans,rotmat,align_modes,align_refcoords,dcoords)
  implicit none
  integer,intent(in) :: ncoord,nvar,dim_renorm
  real(8),intent(in) :: rcoords(nvar) ! the set of coordinates the new ones should be fitted to
  real(8),intent(in) :: xcoords_(nvar),align_modes(nvar,dim_renorm),align_refcoords(nvar)
  real(8),intent(out) :: trans(3),rotmat(3,3)
  real(8),intent(out) :: dcoords(ncoord)
  integer :: nat
  integer:: iat,ivar,jvar
  real(8) :: rmat(3,3),rsmat(3,3),eigvec(3,3),eigval(3)
  real(8) :: center(3)
  real(8) :: weight(nvar)
  real(8) :: xcoords(nvar)
  integer :: itry,i,j
  real(8) :: detrot

  nat=nvar/3
  xcoords=xcoords_

!  if(massweight) then
!    do iat=1,nat
!      weight(iat*3-2:iat*3)=refmass(iat)
!    end do
!  else
    weight=1.D0
!  end if

  ! as compared to dlf_cartesian_align: coords1=rcoords coords2=coords

  trans=0.D0
  rotmat=0.D0
  do ivar=1,3
    rotmat(ivar,ivar)=1.D0
  end do

  ! if there are other cases to ommit a transformation, add them here
!  if(nat==1) return
  !if(.not.superimpose) return

  trans=0.D0
  center=0.D0
  do iat=1,nat
    center(:)=center(:)+rcoords(iat*3-2:iat*3)*weight(iat*3-2:iat*3)
    trans(:)=trans(:)+(xcoords(iat*3-2:iat*3)-rcoords(iat*3-2:iat*3))*weight(iat*3-2:iat*3)
  end do
  trans=trans/sum(weight)*3.D0
  center=center/sum(weight)*3.D0

  !print*,"# trans",trans

  ! translate them to common centre
  do iat=1,nat
    xcoords(iat*3-2:iat*3)=xcoords(iat*3-2:iat*3)-trans(:)
  end do

  rmat=0.D0
  do iat=1,nat
    do ivar=1,3
      do jvar=1,3
        rmat(ivar,jvar)=rmat(ivar,jvar)+weight(3*iat)*(rcoords(ivar+3*iat-3)-center(ivar))* &
            (xcoords(jvar+3*iat-3)-center(jvar))
      end do
    end do
  end do
  rmat=rmat/sum(weight)*3.D0
  !write(*,"('R   ',3f10.3)") rmat
  rsmat=transpose(rmat)
  eigvec=matmul(rsmat,rmat)
  rsmat=eigvec

  !write(stdout,"('RtR ',3f10.3)") rsmat
  call matrix_diagonalise(3,rsmat,eigval,eigvec)

  ! It turns out that the rotation matrix may have a determinat of -1
  ! in the procedure used here, i.e. the system is mirrored - which is
  ! wrong chemically. This can be avoided by inserting a minus in the
  ! equation
  ! 1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))

  ! So, here we first calculate the rotation matrix, and if it is
  ! zero, the first eigenvalue is reversed

  do itry=1,2
    ! rsmat are the vectors b:
    j=-1
    do i=1,3
      if(eigval(i)<1.D-8) then
        if(i>1) then
          ! the system is linear - no rotation necessay.
          ! WHY ?! There should still be one necessary!
          return
          !print*,"Eigenval. zero",i,eigval(i)
          !call dlf_fail("Error in dlf_cartesian_align")
        end if
        j=1
      else
        if(i==1.and.itry==2) then
          rsmat(:,i)=-1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
        else
          rsmat(:,i)=1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
        end if
      end if
    end do
    if(j==1) then
      ! one eigenvalue was zero, the system is planar
      rsmat(1,1)=rsmat(2,2)*rsmat(3,3)-rsmat(3,2)*rsmat(2,3)
      rsmat(2,1)=rsmat(3,2)*rsmat(1,3)-rsmat(1,2)*rsmat(3,3)
      rsmat(3,1)=rsmat(1,2)*rsmat(2,3)-rsmat(2,2)*rsmat(1,3)
      ! deal with negative determinant
      if (itry==2) then
         rsmat(:,1) = -rsmat(:,1)
      end if
    end if

    do i=1,3
      do j=1,3
        rotmat(i,j)=sum(rsmat(i,:)*eigvec(j,:))
      end do
    end do
    !write(*,"('rotmat ',3f10.3)") rotmat
    detrot=   &
        rotmat(1,1)*(rotmat(2,2)*rotmat(3,3)-rotmat(2,3)*rotmat(3,2)) &
        -rotmat(2,1)*(rotmat(1,2)*rotmat(3,3)-rotmat(1,3)*rotmat(3,2)) &
        +rotmat(3,1)*(rotmat(1,2)*rotmat(2,3)-rotmat(1,3)*rotmat(2,2))
    !write(*,*) "Determinat of rotmat", detrot
    if(detrot > 0.D0) exit
    if(detrot < 0.D0 .and. itry==2) then
      stop "Error in dlf_cartesian_align, obtained a mirroring instead of rotation."
    end if

  end do


!!$  do ivar=1,3
!!$    rsmat(:,ivar)=1.d0/dsqrt(eigval(ivar)) * matmul(rmat,eigvec(:,ivar))
!!$  end do
!!$
!!$  do ivar=1,3
!!$    do jvar=1,3
!!$      rotmat(ivar,jvar)=sum(rsmat(ivar,:)*eigvec(jvar,:))
!!$    end do
!!$  end do

  ! transform coordinates
  do iat=1,nat
    xcoords(iat*3-2:iat*3)= xcoords(iat*3-2:iat*3)-center
    xcoords(iat*3-2:iat*3)=matmul(rotmat,xcoords(iat*3-2:iat*3))
    xcoords(iat*3-2:iat*3)= xcoords(iat*3-2:iat*3)+center
  end do

!!$  ! write xyz
!!$  if(ttrain) then
!!$    fid=52
!!$  else
!!$    fid=53
!!$  end if
!!$  write(fid,*) nat
!!$  write(fid,*) 
!!$  do iat=1,nat
!!$    write(fid,'(" H ",3f12.7)') xcoords(iat*3-2:iat*3)*5.2917720810086D-01
!!$  end do
  
!  print*,"transformed coordinates"
!  write(*,'(3f15.5)') coords
  
  ! now all quantities have been transformed to c-coords (or relative to c-coordinates)
  
  ! now, the coordinates need to be mass-weighted!

  dcoords=matmul(transpose(align_modes),sqrt(weight)*(xcoords-align_refcoords))

end subroutine cgh_xtos_simple

subroutine r_diagonal(nm,a,eignum,eigvect)
  implicit integer(i-n)
  implicit real(8)(a-h,o-z)
  real(8) a(nm,nm),eigvect(nm,nm),a_store(nm,nm)
  real(8) VL,VU,ABSTOL,eignum(nm)
  integer, dimension(:),allocatable:: ISUPPZ,IWORK
  character*1 JOBZ,RANGE,UPLO
  real(8), dimension(:),allocatable:: WORK

  a_store=a

  allocate(ISUPPZ(2*nm))
  LWORK=26*nm
  allocate(WORK(LWORK))
  LIWORK=10*nm
  allocate(IWORK(LIWORK))
  JOBZ='V'
  RANGE='A'
  UPLO='U'
  VL=-1.d3
  VU=1.d3
  IL=1
  IU=nm
  ABSTOL=1.d-16

  LWORK=-1
  LIWORK=-1
  call DSYEVR( JOBZ, RANGE, UPLO, nm, A_STORE, nm, VL, VU, IL, IU,&
                    ABSTOL, M, eignum,eigvect, nm, ISUPPZ,& 
                    WORK, LWORK,&
                    IWORK, LIWORK, INFO )
  if(info.ne.0)then      
    write(*,*)'r_diagonal: INFO NON-ZERO',INFO
  endif

  LWORK=idint(WORK(1))
  LIWORK=IWORK(1)
  deallocate(IWORK)
  deallocate(WORK)
  allocate(WORK(LWORK))
  allocate(IWORK(LIWORK))

  call DSYEVR( JOBZ, RANGE, UPLO, nm, A_STORE, nm, VL, VU, IL, IU,&
                    ABSTOL, M, eignum,eigvect, nm, ISUPPZ, &
                    WORK, LWORK,&
                    IWORK, LIWORK, INFO )


  deallocate(ISUPPZ)
  deallocate(WORK)
  deallocate(IWORK)
  return
end subroutine r_diagonal

subroutine interatomic_coord_conversion(nat,ncoord,ns,ntest,x_cart,g_cart,&
h_cart,refcoords,refgrad,refhess,testcoords,testgrad,testhess,&
DMAT,DMAT_PINV,DMAT_PINV2,projection,KMAT,KMAT_Q,DM2,radii_omit,&
pinv_tol,pinv_tol_back,coords_inverse,dim_renorm,xfreqs_store,tol_type,refmass)
  implicit none
  integer, intent(in) :: ns,ntest,ncoord,nat,dim_renorm,tol_type
  real(8), intent(in) :: x_cart(3*nat,ns+ntest),g_cart(3*nat,ns+ntest),&
  h_cart(3*nat,3*nat,ns+ntest),refmass(nat)
  real(8), intent(out) :: refcoords(ncoord,ns),refgrad(ncoord,ns),refhess(ncoord,ncoord,ns),testcoords(ncoord,ntest),&
  testgrad(ncoord,ntest),testhess(ncoord,ncoord,ntest)
  real(8), intent(inout) :: pinv_tol,pinv_tol_back
  real(8), intent(in) :: xfreqs_store(3*nat,ns+ntest)
  logical, intent(in)  :: coords_inverse
  real(8), intent(out) :: DMAT(ncoord,3*nat,ns+ntest),DMAT_PINV(3*nat,ncoord,ns+ntest),&
  DMAT_PINV2(ncoord,3*nat,ns+ntest),projection(ncoord,ncoord,ns+ntest),&
  KMAT(3*nat,3*nat,ns+ntest),KMAT_Q(ncoord,ncoord,ns+ntest),&
  DM2(3*nat,3*nat,ncoord,ns+ntest)
  integer, intent(out) :: radii_omit(3*nat+1)
!  integer redundant_dof
  real(8), allocatable :: x_ia(:,:),g_ia(:,:),h_ia(:,:,:)

!  if(barycentre_switch)then
!    redundant_dof=nat*(nat-1)/2+nat
!  else
!    redundant_dof=nat*(nat-1)/2
!  endif

  allocate(x_ia(ncoord,ns+ntest))
  allocate(g_ia(ncoord,ns+ntest))
  allocate(h_ia(ncoord,ncoord,ns+ntest))
!  call coords_to_interatomic(nat,ncoord,redundant_dof,ns,ntest,refmass,barycentre,x_cart,&
!  g_cart,h_cart,x_ia,g_ia,h_ia,DMAT,DMAT_PINV,DMAT_PINV2,KMAT,coords_inverse,barycentre_switch,&
!  pinv_tol)
  call coords_to_interatomic(nat,ncoord,ns,ntest,x_cart,&
  g_cart,h_cart,x_ia,g_ia,h_ia,DMAT,DMAT_PINV,DMAT_PINV2,projection,KMAT,KMAT_Q,DM2,radii_omit,&
  pinv_tol,pinv_tol_back,coords_inverse,dim_renorm,xfreqs_store,tol_type,refmass)

  refcoords=x_ia(:,1:ns)
  testcoords=x_ia(:,1+ns:)
  deallocate(x_ia)
  refgrad=g_ia(:,1:ns)
  testgrad=g_ia(:,1+ns:)
  deallocate(g_ia)
  refhess=h_ia(:,:,1:ns)
  testhess=h_ia(:,:,ns+1:)
  deallocate(h_ia)

end subroutine interatomic_coord_conversion

subroutine redundant_IAcoords(nat,interatomic_dof,ns,ntest,x_unchanged,x_out,x_out_mw,x_out_nmw,&
coords_inverse,refmass)
  implicit none
  integer, intent(in) :: ns,ntest,nat,interatomic_dof
  real(8), intent(in) :: x_unchanged(3*nat,ns+ntest),refmass(nat)
  real(8), intent(out):: x_out(interatomic_dof,ns+ntest),x_out_mw(interatomic_dof,ns+ntest),&
  x_out_nmw(interatomic_dof,ns+ntest)
  logical, intent(in) :: coords_inverse
  integer inv_fac,counter,i,j,k

  if(coords_inverse)inv_fac=-1
  if(.not.coords_inverse)inv_fac=1
  
  counter=0
  do i=1,nat
    do j=i+1,nat
      counter=counter+1
      do k=1,ns+ntest
        x_out(counter,k)=dsqrt(sum((x_unchanged(3*i-2:3*i,k)-&
        x_unchanged(3*j-2:3*j,k))**2,dim=1))**inv_fac
        x_out_mw(counter,k)=dsqrt(sum((x_unchanged(3*i-2:3*i,k)/sqrt(refmass(i))-&
        x_unchanged(3*j-2:3*j,k)/sqrt(refmass(j)))**2,dim=1))**inv_fac
        x_out_nmw(counter,k)=dsqrt(sum((x_unchanged(3*i-2:3*i,k)*sqrt(refmass(i))-&
        x_unchanged(3*j-2:3*j,k)*sqrt(refmass(j)))**2,dim=1))**inv_fac
      enddo
    enddo
  enddo
  x_out_nmw=x_out

end subroutine redundant_IAcoords

subroutine load_ia_coords(i_red,nat,radii_omit,ns,ntest,x_unchanged,g_unchanged,h_unchanged,&
  coords_inverse,ncoord,trans_coordsnr,x_out,pinv_tol,refmass)
  implicit none
  integer, intent(out) :: i_red
  integer, intent(in)  :: nat,ns,ntest,ncoord
  integer, intent(inout) :: radii_omit(3*nat+1)
  real(8), intent(in)  :: x_unchanged(3*nat,ns+ntest),g_unchanged(3*nat,ns+ntest),&
  h_unchanged(3*nat,3*nat,ns+ntest),pinv_tol,refmass(nat)
  logical, intent(in)  :: coords_inverse
  real(8), intent(out) :: trans_coordsnr(ncoord,ncoord,2),x_out(ncoord,ns+ntest)
  real(8), allocatable :: x_outr(:,:),x_outr_mw(:,:),x_outr_nmw(:,:),trans_coordsr(:,:)
  integer, allocatable :: bonds(:,:)
  integer ll,i,j,k,l,kk

  i_red=nat*(nat-1)/2
  allocate(x_outr(i_red,ns+ntest))
  radii_omit(1)=i_red-ncoord
  allocate(x_outr_mw(i_red,ns+ntest))
  allocate(x_outr_nmw(i_red,ns+ntest))
  call redundant_IAcoords(nat,i_red,ns,ntest,x_unchanged,x_outr,x_outr_mw,x_outr_nmw,&
  coords_inverse,refmass)
  deallocate(x_outr_nmw)

  allocate(trans_coordsr(i_red,i_red))
  trans_coordsr=0.d0
  ll=0
  do i=nat,1,-1
    do k=1,i-1
      do l=1,i-1
        if(k+l.ne.i)then
          trans_coordsr(ll+l,ll+k)=1.d0/dble(i-2)
        endif
      enddo
    enddo
    ll=ll+i-1
  enddo
  trans_coordsr=0.d0
  do i=1,i_red
    trans_coordsr(i,i)=1.d0
  enddo
  allocate(bonds(i_red-ncoord,2))
  call non_redundant_smallest_distances(nat,ncoord,i_red,ns,ntest,x_outr,x_out,bonds(:,2),coords_inverse)
  ll=0
  do i=1,i_red
    if(.not.any(bonds(:,2)==i))then
      ll=ll+1
      kk=0
      do j=1,i_red
        if(.not.any(bonds(:,2)==j))then
          kk=kk+1
          trans_coordsnr(ll,kk,1)=trans_coordsr(i,j)
        endif
      enddo
    endif
  enddo

  call make_nr_set(nat,ncoord,i_red,radii_omit,bonds,trans_coordsnr,trans_coordsr,ns,ntest,&
  x_outr,x_unchanged,g_unchanged,h_unchanged,x_out,pinv_tol,coords_inverse)
  deallocate(x_outr_mw)
  deallocate(x_outr)
  deallocate(trans_coordsr)
  radii_omit(2:1+i_red-ncoord)=bonds(:,2)
  deallocate(bonds)

end subroutine load_ia_coords

subroutine make_nr_set(nat,ncoord,i_red,radii_omit,bonds,trans_coordsnr,trans_coordsr,ns,ntest,&
x_outr,x_unchanged,g_unchanged,h_unchanged,x_out,pinv_tol,coords_inverse)
  USE OMP_LIB
  implicit none
  integer, intent(in) :: nat,i_red,ncoord,ns,ntest
  integer, intent(inout) :: radii_omit(3*nat+1)
  integer, intent(inout) :: bonds(i_red-ncoord,2)
  real(8), intent(out) :: trans_coordsnr(ncoord,ncoord,2)
  real(8), intent(in) :: x_outr(i_red,ns+ntest),x_unchanged(3*nat,ns+ntest),g_unchanged(3*nat,ns+ntest),&
  h_unchanged(3*nat,3*nat,ns+ntest),trans_coordsr(i_red,i_red),pinv_tol
  logical, intent(in) :: coords_inverse
  real(8), intent(out) :: x_out(ncoord,ns+ntest)
  integer, allocatable :: bond_atom_count(:),dof_shuffle(:),dof_shuffle_store(:)
  real(8), allocatable :: x_0(:,:)
  logical mtc
  integer l,kk,ll,j,k1,k,i
  integer(8) incr
  real(8) rp(4),ro,ncr
  character(1) check_type
  character(20) progress
  call random_seed()
  allocate(bond_atom_count(nat))
  allocate(dof_shuffle(nat*(nat-1)/2))
  allocate(dof_shuffle_store(nat*(nat-1)/2))
  dof_shuffle_store=(/(i, i=1,nat*(nat-1)/2)/)
  dof_shuffle=dof_shuffle_store
  allocate(x_0(ncoord,ns+ntest))
  mtc=.false.
  l=0
  kk=0
  rp(1:2)=-huge(1.d0)
  rp(3:4)=huge(1.d0)
  check_type='D'
!!*MANUAL REMOVAL OF BONDS
!  bonds(:,2)=(/1,5,10,11,19,20/)!(/4,5,9,10,17,20/)!(/2,5,6,9,17,20/)!(/1,2,4,7,19,20/)!(/1,5,6,11,14,20/)!
!  call DMAT_NR_singularity_check(trans_coordsnr(:,:,1),bonds(:,2),radii_omit(1),nat,i_red,ns,&
!  ntest,x_outr,x_outr_mw,x_out,x_unchanged,g_unchanged,h_unchanged,refmass,coords_inverse,rp(1),rp(3),check_type)
!  trans_coordsnr(:,:,2)=trans_coordsnr(:,:,1)
!!************************
  incr=nint(min(2.d5,exp(ncr(i_red,radii_omit(1)))))

  if(check_type.eq.'D' .and. incr.ne.nint(exp(ncr(i_red,radii_omit(1)))))then
    incr=incr*OMP_GET_NUM_PROCS()*nint(log(max(exp(1.d0),dble(OMP_GET_NUM_PROCS()))))
  else if(check_type.ne.'D' .and. incr.ne.nint(exp(ncr(i_red,radii_omit(1)))))then
    incr=incr*OMP_GET_NUM_PROCS()
  endif
  print*,'LOG(#BOND COMBINATIONS) & #BOND COMBINATIONS TO BE TESTED'
  print*,ncr(i_red,radii_omit(1)),incr
  do k=1,incr
    write(progress,'(F20.13)')100.d0*dble(k)/dble(incr)
    IF(MOD(k,5).EQ.0) WRITE(*,42,ADVANCE='NO') k, ' of ', incr ,CHAR(13)
42  FORMAT(I20,A,I20,A)
    if(nint(exp(ncr(i_red,radii_omit(1)))).eq.incr)then
      call nexksb(i_red,radii_omit(1),bonds(:,1),mtc,l,kk)
    else
      call shuffle(dof_shuffle,nat*(nat-1)/2)
      bonds(:,1)=dof_shuffle(1:radii_omit(1))
      dof_shuffle=dof_shuffle_store
    endif
    j=0
    bond_atom_count=0
    do i=1,nat
      do ll=i+1,nat
        j=j+1
        if(any(bonds(:,1)==j))then
          bond_atom_count(i)=bond_atom_count(i)+1
          bond_atom_count(ll)=bond_atom_count(ll)+1
        endif
      enddo
    enddo
    if(.not.any(bond_atom_count.ge.max(1,nat-3)))then
      ll=0
      do i=1,i_red
        if(.not.any(bonds(:,1)==i))then
          ll=ll+1
          k1=0
          do j=1,i_red
            if(.not.any(bonds(:,1)==j))then
              k1=k1+1
              trans_coordsnr(ll,k1,1)=trans_coordsr(i,j)
            endif
          enddo
        endif
      enddo
      call DMAT_NR_singularity_check(trans_coordsnr(:,:,1),bonds(:,1),radii_omit(1),nat,i_red,ns,&
      ntest,x_outr,x_0,x_unchanged,g_unchanged,h_unchanged,rp(1),rp(3),check_type,pinv_tol,coords_inverse)
      if(check_type.eq.'S')then
        if(rp(1).gt.rp(2))then
          rp(2)=rp(1)
          bonds(:,2)=bonds(:,1)
          trans_coordsnr(:,:,2)=trans_coordsnr(:,:,1)
          x_out=x_0
          print*,'SINGULARITY',rp(2)
        endif
!      if(rp(2).gt.1.d-15)exit
      elseif(check_type.eq.'D'.or.check_type.eq.'V'.or.check_type.eq.'G'.or.check_type.eq.'H')then
        if(rp(3).lt.rp(4))then
          call DMAT_NR_singularity_check(trans_coordsnr(:,:,1),bonds(:,1),radii_omit(1),nat,i_red,ns,&
          ntest,x_outr,x_0,x_unchanged,g_unchanged,h_unchanged,rp(1),ro,'S',pinv_tol,coords_inverse)
          if(rp(1).gt.1.d-6)then
            rp(4)=rp(3)
            bonds(:,2)=bonds(:,1)
            trans_coordsnr(:,:,2)=trans_coordsnr(:,:,1)
            x_out=x_0
            print*,'DIST/VAR/GRAD/HESS',rp(3),rp(1)
          endif
        endif
      else
        print*,'NO VALID CHECK GIVEN'
        stop
      endif
    endif
!    if(bonds(1,1).eq.i_red-radii_omit(1) .and.bonds(radii_omit(1),1).eq.i_red)then
!      print*,'REACHED LAST SET OF BONDS',bonds(:,1)
!      exit
!    endif
  enddo

  deallocate(x_0)
  deallocate(dof_shuffle)
  deallocate(dof_shuffle_store)
  deallocate(bond_atom_count)

end subroutine make_nr_set

subroutine coords_to_interatomic(nat,ncoord,ns,ntest,x_unchanged,g_unchanged,h_unchanged,&
x_out,g_out,h_out,DMAT,DMAT_PINV,DMAT_PINV2,projection,KMAT,KMAT_Q,DM2,radii_omit,pinv_tol,pinv_tol_back,coords_inverse,&
dim_renorm,xfreqs_store,tol_type,refmass)
  USE OMP_LIB
  implicit none
  integer, intent(in) :: ntest,ns,nat,ncoord,dim_renorm,tol_type
  real(8), intent(in) :: x_unchanged(3*nat,ns+ntest),g_unchanged(3*nat,ns+ntest),h_unchanged(3*nat,3*nat,ns+ntest),&
  xfreqs_store(3*nat,ns+ntest),refmass(nat)
  real(8), intent(inout) :: pinv_tol,pinv_tol_back
  logical, intent(in) :: coords_inverse
  real(8), intent(out) :: x_out(ncoord,ns+ntest),g_out(ncoord,ns+ntest),&
  h_out(ncoord,ncoord,ns+ntest),DMAT(ncoord,3*nat,ns+ntest),&
  DMAT_PINV(3*nat,ncoord,ns+ntest),DMAT_PINV2(ncoord,3*nat,ns+ntest),KMAT(3*nat,3*nat,ns+ntest),&
  DM2(3*nat,3*nat,ncoord,ns+ntest),projection(ncoord,ncoord,ns+ntest),&
  KMAT_Q(ncoord,ncoord,ns+ntest)
  integer, intent(out) ::radii_omit(3*nat+1)
  real(8), allocatable :: R_DM2(:,:,:),LL1(:,:),LL2(:,:),&
  x_0(:,:),delta_x(:),delta_rr(:),delta_rnr(:),h_u(:,:),&
  trans_coordsnr(:,:,:)
  integer i,k,l,kk,i_red,TID,p1,p2,nzero!inter_bond_dependencies
  integer, allocatable :: sortlist(:)!,map(:)
  real(8) tol_save_res,tol_save_x,tol_save_resx,tol_save_graphical,tol,rp(6),ro(3),tol_back,&
  pi
  real(8), allocatable :: KMAT_COMP(:,:,:),evals(:,:),storage1(:,:),storage2(:,:)

  pi=4.d0*atan(1.d0)
  nzero=3*nat-dim_renorm
  allocate(trans_coordsnr(ncoord,ncoord,2))
  call load_ia_coords(i_red,nat,radii_omit,ns,ntest,x_unchanged,g_unchanged,h_unchanged,coords_inverse,&
  ncoord,trans_coordsnr,x_out,pinv_tol,refmass)
  p1=1
  p2=1

  DMAT_PINV=0.d0
  h_out=0.d0
  g_out=0.d0
  allocate(R_DM2(3*nat,3*nat,ncoord))
  KMAT=0.d0

  call make_DMAT(nat,i_red,ncoord,ns,ntest,DMAT,x_out,x_unchanged,radii_omit(p1:p2),coords_inverse)
  do k=1,ns+ntest
    DMAT(:,:,k)=matmul(trans_coordsnr(:,:,2),DMAT(:,:,k))
    x_out(:,k)=matmul(trans_coordsnr(:,:,2),x_out(:,k))
  enddo

  p1=ncoord
  p2=3*nat
  allocate(LL1(p1,ncoord))
  allocate(LL2(p2,3*nat))
  LL1=0.d0
  LL2=0.d0
  do i=1,min(ncoord,p1)
    LL1(i,i)=1.d0
  enddo
  do i=1,min(3*nat,p2)
    LL2(i,i)=1.d0
  enddo
  allocate(x_0(ncoord,ncoord))
  allocate(delta_rr(i_red))
  allocate(delta_rnr(ncoord))
  allocate(delta_x(3*nat))
  allocate(KMAT_COMP(3*nat,3*nat,4))
  allocate(storage1(3*nat,3*nat-nzero))
  allocate(storage2(3*nat-nzero,3*nat-nzero))
  allocate(evals(3*nat,2))
  allocate(sortlist(1+nzero))
  allocate(h_u(3*nat,3*nat))
  tol=0.d0
  rp=0.d0
  ro=0.d0
  rp(5)=minval(xfreqs_store(1,:))
!!$OMP PARALLEL DO &
!!$OMP PRIVATE(k,R_DM2,l,kk,i,delta_x,delta_rr,delta_rnr,tol_save_res,tol_save_x,&
!!$OMP tol_save_resx,tol_save_graphical,x_0,TID,tol,tol_back,KMAT_COMP,evals,sortlist,&
!!$OMP storage1,storage2)&
!!$OMP SHARED(ns,nat,ntest,nzero,DMAT,g_unchanged,ncoord,x_out,x_unchanged,h_unchanged,&
!!$OMP coords_inverse,DM2,DMAT_PINV2,g_out,DMAT_PINV,KMAT,h_out,projection,&
!!$OMP i_red,radii_omit,pinv_tol,bond_depends,trans_coordsnr,rp,LL1,LL2,p1,p2,&
!!$OMP eval_ave,eval_stddev,h_u), default(none)
  do k=1,ns+ntest
    x_0=0.d0
    TID=0!    TID = OMP_GET_THREAD_NUM()
    call DDMAT(nat,ncoord,x_out(:,k),DMAT(:,:,k),R_DM2,coords_inverse)
    call r_diagonal(3*nat,h_unchanged(:,:,k),evals(:,2),KMAT_COMP(:,:,1))
    DM2(:,:,:,k)=R_DM2

!    call tolerance_simple_hess(transpose(DMAT(:,:,k)),x_0,h_unchanged(:,:,k),g_unchanged(:,k),&
!    R_DM2,3*nat,ncoord,nzero,tol_save_res,tol_save_x,tol_save_resx,tol_save_graphical,'M',TID)
    call tolerance_simple(transpose(DMAT(:,:,k)),x_0(:,1),g_unchanged(:,k),3*nat,ncoord,tol_save_res,tol_save_x,&
    tol_save_resx,tol_save_graphical,'M',TID)
    print*,k,'TOLERANCE',tol_save_res,tol_save_x,tol_save_graphical
    rp(1)=rp(1)+tol_save_x
    rp(2)=rp(2)+tol_save_res
    rp(3)=rp(3)+tol_save_graphical
    if(tol_type==1)then
      tol=tol_save_res
    elseif(tol_type==2)then
      tol=tol_save_graphical
    elseif(tol_type==3)then
      tol=pinv_tol
    else
      print*,'NO VALID TOLERANCE TYPE GIVEN IN inp.dat'
    endif

    CALL Pseudoinverse(3*nat,ncoord,transpose(DMAT(:,:,k)),DMAT_PINV2(:,:,k),tol,'M')
    x_0(:,1)=matmul(DMAT_PINV2(:,:,k),g_unchanged(:,k))

    call vary_vector(x_unchanged(:,k),nat,delta_x,delta_rr,i_red)
    kk=0
    do l=1,i_red
      if(.not.any(radii_omit(2:1+i_red-ncoord)==l))then
        kk=kk+1
        delta_rnr(kk)=delta_rr(l)
      endif
    enddo
    delta_rnr=matmul(trans_coordsnr(:,:,2),delta_rnr)

    call tolerance_simple(DMAT(:,:,k),delta_x,delta_rnr,ncoord,3*nat,tol_save_res,tol_save_x,&
    tol_save_resx,tol_save_graphical,'M',TID)
    print*,k,'REVERSE TOLERANCE',tol_save_res,tol_save_x,tol_save_graphical

    ro(1)=ro(1)+tol_save_x
    ro(2)=ro(2)+tol_save_res
    ro(3)=ro(3)+tol_save_graphical
    if(tol_type==1)then
      tol_back=tol_save_res
    elseif(tol_type==2)then
      tol_back=tol_save_graphical
    elseif(tol_type==3)then
      tol_back=pinv_tol_back
    else
      print*,'NO VALID TOLERANCE TYPE GIVEN IN inp.dat'
    endif

    CALL Pseudoinverse(ncoord,3*nat,DMAT(:,:,k),DMAT_PINV(:,:,k),tol_back,'M')
    projection(:,:,k)=matmul(DMAT(:,:,k),DMAT_PINV(:,:,k))
    projection(:,:,k)=0.d0
    do l=1,ncoord
      projection(l,l,k)=1.d0
    enddo

    g_out(:,k)=matmul(projection(:,:,k),x_0(:,1))
    rp(4)=rp(4)+dot_product(g_unchanged(:,k),matmul(transpose(DMAT(:,:,k)),g_out(:,k)))/&
    sqrt(sum(g_unchanged(:,k)**2)*sum(matmul(transpose(DMAT(:,:,k)),g_out(:,k))**2))
    if(sum(g_unchanged(:,k)**2).gt.1.d-15)print*,k,'COHERENCE        ',&
    dot_product(g_unchanged(:,k),matmul(transpose(DMAT(:,:,k)),g_out(:,k)))/&
    sqrt(sum(g_unchanged(:,k)**2)*sum(matmul(transpose(DMAT(:,:,k)),g_out(:,k))**2))

    do l=1,3*nat
      do kk=1,3*nat
        do i=1,ncoord
          KMAT(l,kk,k)=KMAT(l,kk,k)+x_0(i,1)*R_DM2(l,kk,i)
        enddo
      enddo
    enddo

    KMAT_Q(:,:,k)=matmul(matmul(DMAT_PINV2(:,:,k),KMAT(:,:,k)),&
    DMAT_PINV(:,:,k))

!    h_out(:,:,k)=matmul(matmul(DMAT_PINV2(:,:,k),h_unchanged(:,:,k)-KMAT(:,:,k)),DMAT_PINV(:,:,k))
    h_out(:,:,k)=matmul(matmul(DMAT_PINV2(:,:,k),h_unchanged(:,:,k)),DMAT_PINV(:,:,k))-KMAT_Q(:,:,k)
    h_out(:,:,k)=matmul(matmul(projection(:,:,k),h_out(:,:,k)),projection(:,:,k))
    rp(6)=rp(6)+dot_product(reshape(h_unchanged(:,:,k),(/3*nat*3*nat/)),&
    reshape(matmul(matmul(transpose(DMAT(:,:,k)),h_out(:,:,k)),DMAT(:,:,k))+KMAT(:,:,k),(/3*nat*3*nat/)))/&
    sqrt(sum(h_unchanged(:,:,k)**2)*sum((matmul(matmul(transpose(DMAT(:,:,k)),h_out(:,:,k)),DMAT(:,:,k))+KMAT(:,:,k))**2))

  enddo

!!$OMP END PARALLEL DO
  rp=rp/dble(ns+ntest)
  ro=ro/dble(ns+ntest)
  print*,'AVE TOL FORWARD, x:', rp(1),'res:',rp(2),'graphical:',rp(3),'COHERENCE',rp(4),'HESS COHERENCE',rp(6)
  print*,'AVE TOL BACK   , x:', ro(1),'res:',ro(2),'graphical:',ro(3)

  if(tol_type==1)then
    pinv_tol=rp(2)
    pinv_tol_back=ro(2)
  elseif(tol_type==2)then
    pinv_tol=rp(3)
    pinv_tol_back=ro(3)
  elseif(tol_type .lt. 1 .or. tol_type .ge.4)then
    print*,'NO VALID TOLERANCE TYPE GIVEN IN inp.dat'
  endif

  deallocate(trans_coordsnr)
  deallocate(KMAT_COMP)
  deallocate(storage1)
  deallocate(storage2)
  deallocate(h_u)
  deallocate(sortlist)
  deallocate(evals)
  deallocate(x_0)
  deallocate(delta_x)
  deallocate(delta_rr)
  deallocate(delta_rnr)
  deallocate(R_DM2)
  deallocate(LL1)
  deallocate(LL2)

end subroutine coords_to_interatomic

subroutine make_DMAT(nat,i_red,ncoord,ns,ntest,DMAT,x_out,x_unchanged,omit_bonds,coords_inverse)
  implicit none
  logical, intent(in) :: coords_inverse
  integer, intent(in) :: ns,ntest,nat,ncoord,i_red,omit_bonds(i_red-ncoord)
  real(8), intent(out):: DMAT(ncoord,3*nat,ns+ntest)
  real(8), intent(in) :: x_unchanged(3*nat,ns+ntest),x_out(ncoord,ns+ntest)
  real(8), allocatable :: vector_diff(:,:)
  integer i,j,k,m,l,counter
  integer, allocatable :: mapping(:,:)

  allocate(vector_diff(3*nat,nat))
  allocate(mapping(nat,nat))
  mapping=-1
  counter=0
  k=0
  do i=1,nat
    do j=i+1,nat
      counter=counter+1
      if(.not.any(omit_bonds==counter))then
        k=k+1
        mapping(i,j)=k
        mapping(j,i)=k
      endif
    enddo
  enddo
  DMAT=0.d0
  do k=1,ns+ntest
    vector_diff=0.d0
    do m=1,nat
      do l=1,nat
        vector_diff(3*l-2:3*l,m)=x_unchanged(3*m-2:3*m,k)-&
        x_unchanged(3*l-2:3*l,k)
      enddo
    enddo

    do m=1,nat
      do l=m+1,nat
        if(mapping(l,m).ne.-1)then
          if(coords_inverse)then
            if(abs(x_out(mapping(l,m),k)) .lt. 1.d3)then
              DMAT(mapping(l,m),3*m-2:3*m,k)=-vector_diff(3*l-2:3*l,m)*&
              x_out(mapping(l,m),k)**3
!                x_out(mapping(l,m),k)**2*2.d0
            endif
          else
            DMAT(mapping(l,m),3*m-2:3*m,k)=vector_diff(3*l-2:3*l,m)/&
            x_out(mapping(l,m),k)
          endif
          DMAT(mapping(l,m),3*l-2:3*l,k)=-DMAT(mapping(l,m),3*m-2:3*m,k)
        endif
      enddo
    enddo
  enddo
  deallocate(mapping)
  deallocate(vector_diff)

end subroutine make_DMAT

subroutine DDMAT(nat,ncoord,x_out,DMAT,DM2,coords_inverse)
  implicit none
  logical, intent(in) :: coords_inverse
  integer, intent(in) :: nat,ncoord
  real(8), intent(in) :: x_out(ncoord),DMAT(ncoord,3*nat)
  real(8), intent(out):: DM2(3*nat,3*nat,ncoord)
  real(8) rp,kronecker
  integer i,l,ata,kk,atb,j

  DM2=0.d0
  do i=1,ncoord
    do l=1,3*nat
      ata=(l+2)/3
      do kk=1,3*nat
        atb=(kk+2)/3
        rp=0.d0
        do j=1,3
          rp=rp+(kronecker(3*ata-3+j,l)-kronecker(3*atb-3+j,l))*&
          (kronecker(3*ata-3+j,kk)-kronecker(3*atb-3+j,kk))
        enddo
        if(coords_inverse)then
          if(abs(x_out(i)).gt.1.d-9 .and. abs(x_out(i)) .lt. 1.d3)then
            DM2(l,kk,i)=(3.d0/x_out(i))*DMAT(i,l)*DMAT(i,kk)-x_out(i)**3*rp
!              DM2(l,kk,i)=(2.d0/x_out(i))*DMAT(i,l)*DMAT(i,kk)-x_out(i)**2*rp*2.d0
          endif
        else
          if(abs(x_out(i)).gt.1.d-9)then
            DM2(l,kk,i)=(1.d0/x_out(i))*(-DMAT(i,l)*DMAT(i,kk)+rp)
          endif
        endif
      enddo
    enddo
  enddo

end subroutine DDMAT

subroutine tolerance_simple(A,x_0,b,m,n,tol_save_res,tol_save_x,tol_save_resx,tol_save_graphical,opt,TID)
  implicit none
  integer, intent(in) :: m,n,TID
  real(8), intent(in) :: A(m,n),b(m)
  real(8), intent(inout):: x_0(n,1)
  real(8), intent(out):: tol_save_res,tol_save_x,tol_save_resx,tol_save_graphical
  character(1), intent(in) :: opt
  integer K,i,j,l
  real(8), allocatable :: a_in(:,:),u(:,:),s(:),vt(:,:),s_store(:,:),&
  L_curve_dot_product(:),tols(:),L_curve(:,:)
  real(8) mod_xlambda(2),mod_residue(2),tol_init,x_lambda(n,1),bb(m,1),&
  range_factor,range_factor_init,residue(m,1),tol_guess,mod_resx(2),&
  l_curve_dist,oldxy(2),newxy(2),oldvec(2,2),newvec(2,3),svar,mod_dotprod!,stddev
  logical file_exists,check_vec_zero
  character(1) CTID
  character(2) KSVD

  check_vec_zero=.false.
  if(sum(x_0).eq.0.d0)then
    check_vec_zero=.true.
  endif

  write(CTID,'(I1.1)')TID

  inquire(file='L_curve_'//CTID//'.dat',exist=file_exists)
  if(file_exists)then
    open(file='L_curve_'//CTID//'.dat',unit=91+TID)
  else
    open(file='L_curve_'//CTID//'.dat',unit=91+TID,status='new')
  endif
  inquire(file='L_curve_D_'//CTID//'.dat',exist=file_exists)
  if(file_exists)then
    open(file='L_curve_D_'//CTID//'.dat',unit=192+TID)
  else
    open(file='L_curve_D_'//CTID//'.dat',unit=192+TID,status='new')
  endif

  range_factor_init=1.4d0
  allocate(a_in(m,n))
  a_in=a
  K = MIN(M,N)
  bb(:,1)=b

  allocate(u(m,K))
  allocate(s(K))
  write(KSVD,'(I2.2)')K
  allocate(vt(K,n))
  allocate(s_store(K,K))
  s_store=0.d0
  call svd(m,n,K,a_in,s,u,vt)
  print*,'SINGULAR VALUES'
  write(*,'('//KSVD//'E11.4)')s
  tol_init=max(1.d-2*minval(abs(s)),1.d-8*maxval(abs(s)))
  tol_guess=tol_init
  tol_save_res=tol_guess
  tol_save_resx=tol_guess
  tol_save_x=tol_guess
  tol_save_graphical=tol_guess
  mod_dotprod=1.d0
  mod_residue(1)=huge(1.d0)
  mod_xlambda(1)=huge(1.d0)
  mod_resx(1)=huge(1.d0)
  if(opt.eq.'T')then
    do i=1,K
      s_store(i,i)=s(i)/(s(i)**2+tol_guess**2)
    enddo
  !  do i=1,k
  !    if(dabs(s(i)).gt.tol_guess)then
  !      s_store(i,i)=1.d0/s(i)
  !    endif
  !  enddo
    j=0
    l=0
  !  do j=2,K-1
    do while(tol_guess.lt.100.d0*maxval(abs(s)))
      j=j+1
      range_factor=range_factor_init**sqrt(dble(j))
      x_lambda=matmul(matmul(matmul(transpose(vt),s_store),transpose(u)),bb)
      residue=matmul(a,x_lambda)-bb
      if(check_vec_zero)x_0=sum(x_lambda)/dble(n)
      mod_xlambda(2)=log(sum((x_lambda-x_0)**2))
      mod_residue(2)=log(sum(residue(:,1)**2))
      mod_resx(2)=mod_xlambda(2)+mod_residue(2)
      newxy=(/mod_xlambda(2),mod_residue(2)/)

      write(91+TID,'(E15.8,X,E15.8,X,E15.8)')tol_guess,mod_residue(2),mod_xlambda(2)
!      if(l.ge.2)then
!        write(92,'(E15.8,X,E15.8)')mod_xlambda(2),dot_product(oldvec(:,1),newvec(:,1))!,newvec(:,2),newvec(:,3),oldvec(:,2)
!      endif
!      if(l.ge.1)then
!        oldvec=newvec(:,1:2)
!      endif
      if(j.eq.1)then
        oldxy=newxy
      endif
      if(j.gt.1)then
        l_curve_dist=sqrt(sum((newxy-oldxy)**2))
        if(l_curve_dist.gt.1.d-2)then
          l=l+1
          newvec(:,1)=newxy-oldxy
          newvec(:,2)=newxy
          newvec(:,3)=oldxy
          if(l.ge.2)then
            write(192+TID,'(E15.8,X,E15.8)')tol_guess,dot_product(oldvec(:,1),newvec(:,1))/&
            sqrt(sum(oldvec(:,1)**2)*sum(newvec(:,1)**2))
          endif
          oldvec=newvec(:,1:2)
          oldxy=newxy
        endif
      endif
!      print*,tol_guess,mod_residue
      if(mod_xlambda(2).lt.mod_xlambda(1))then
        mod_xlambda(1)=mod_xlambda(2)
        tol_save_x=tol_guess
      endif
      if(mod_residue(2).lt.mod_residue(1))then
        mod_residue(1)=mod_residue(2)
        tol_save_res=tol_guess
      endif
      if(mod_resx(2).lt.mod_resx(1))then
        mod_resx(1)=mod_resx(2)
        tol_save_resx=tol_guess
      endif
      tol_guess=tol_guess+range_factor*tol_init!/dble(steps)
  !    tol_guess=dabs(s(j))
      s_store=0.d0
      do i=1,K
        s_store(i,i)=s(i)/(s(i)**2+tol_guess**2)
      enddo
  !    do i=1,k
  !      if(dabs(s(i)).gt.tol_guess)then
  !        s_store(i,i)=1.d0/s(i)
  !      endif
  !    enddo
    enddo
    close(91+TID)
    close(192+TID)
    open(file='L_curve_D_'//CTID//'.dat',unit=192+TID, status='old')
    open(file='L_curve_'//CTID//'.dat',unit=91+TID, status='old')
    allocate(L_curve(j-1,4))
    do i=1,j-1
      read(91+TID,'(E15.8,X,E15.8,X,E15.8)')L_curve(i,1),L_curve(i,3),L_curve(i,2)
    enddo
!!   FIND AVERAGE AND OF 2*LOG|X|
!    svar=sum(L_curve(:,2))/dble(j-1)
!    stddev=sqrt(sum((L_curve(:,2)-svar)**2)/dble(j-1))
!!   SHIFT 2*LOG|X| TO POSITIVE VALUES
!    L_curve(:,2)=L_curve(:,2)-svar+2.d0*stddev
!!   FIND AVERAGE AND OF 2*LOG|A*X-B|
!    svar=sum(L_curve(:,3))/dble(j-1)
!    stddev=sqrt(sum((L_curve(:,3)-svar)**2)/dble(j-1))
!!   SHIFT 2*LOG|A*X-B| TO POSITIVE VALUES
!    L_curve(:,3)=L_curve(:,3)-svar+2.d0*stddev

    svar=minval(L_curve(:,2))
    L_curve(:,2)=L_curve(:,2)-svar
    svar=minval(L_curve(:,3))
    L_curve(:,3)=L_curve(:,3)-svar

!    open(file='L_curve.dat',unit=91, status='replace')
!    do i=1,j-1
!      write(91,'(E15.8,X,E15.8,X,E15.8)')L_curve(i,1),L_curve(i,3),L_curve(i,2)
!    enddo
!    close(91)
!   FIND DISTANCE OF L_CURVE TO THE ORIGIN, LOCALISE SMALLEST DISTANCE (SHOULD BE THE CORNER)
    L_curve(:,4)=sqrt(l_curve(:,2)**2+l_curve(:,3)**2)
    i=minloc(L_curve(:,4),dim=1)
    tol_save_graphical=l_curve(i,1)
    write(91+TID,'(A,X,4E15.8)')'#',tol_save_res,tol_save_x,tol_save_resx,tol_save_graphical
    close(91+TID)
    deallocate(L_curve)
    allocate(L_curve_dot_product(l-1))
    allocate(tols(l-1))
    do i=1,l-1
      read(192+TID,'(E15.8,X,E15.8)')tols(i),L_curve_dot_product(i)
    enddo
    close(192+TID)!,status='delete')
    do i=11,l-1
      if(any(L_curve_dot_product(i-10:i-1).lt.0.9999d0))then
        continue
      else
        if(L_curve_dot_product(i).lt.0.9d0*mod_dotprod)then
          mod_dotprod=L_curve_dot_product(i)
!          tol_save_graphical=tols(i)
          L_curve_dot_product(i)=1.d0
          if(all(L_curve_dot_product(i+1:i+10).gt.1.d0-1.d-4))then
            exit
          endif
        endif
      endif
    enddo
  elseif(opt.eq.'M')then
!    do i=1,K
!      s_store(i,i)=s(i)/(s(i)**2+tol_guess**2)
!    enddo
    do i=1,k
      if(dabs(s(i)).gt.tol_guess)then
        s_store(i,i)=1.d0/s(i)
      else
        s_store(i,i)=1.d0/maxval(abs(s)*dble(k))
      endif
    enddo
!    j=0
    do j=K-1,1,-1
!    do while(tol_guess.lt.5.d0*maxval(abs(s)))
!      j=j+1
!      range_factor=range_factor_init**sqrt(dble(j))

      x_lambda=matmul(matmul(matmul(transpose(vt),s_store),transpose(u)),bb)
      residue=matmul(a,x_lambda)-bb
      if(check_vec_zero)x_0=sum(x_lambda)/dble(n)
      mod_xlambda(2)=log(sum((x_lambda-x_0)**2))
      mod_residue(2)=log(sum(residue(:,1)**2))
      mod_resx(2)=mod_xlambda(2)+mod_residue(2)
      write(91+TID,'(E15.8,X,E15.8,X,E15.8)')tol_guess,mod_residue(2),mod_xlambda(2)
      if(mod_xlambda(2).lt.mod_xlambda(1))then
        mod_xlambda(1)=mod_xlambda(2)
        tol_save_x=tol_guess
      endif
      if(mod_residue(2).lt.mod_residue(1))then
        mod_residue(1)=mod_residue(2)
        tol_save_res=tol_guess
      endif
      if(mod_resx(2).lt.mod_resx(1))then
        mod_resx(1)=mod_resx(2)
        tol_save_resx=tol_guess
      endif
!      tol_guess=tol_guess+range_factor*tol_init!/dble(steps)
      tol_guess=dabs(s(j))
      s_store=0.d0
!      do i=1,K
!        s_store(i,i)=s(i)/(s(i)**2+tol_guess**2)
!      enddo
      do i=1,k
        if(dabs(s(i)).gt.tol_guess)then
          s_store(i,i)=1.d0/s(i)
        else
          s_store(i,i)=1.d0/(maxval(abs(s))*dble(K))
        endif
      enddo
    enddo
    close(91+TID)
    close(192+TID)
    open(file='L_curve_'//CTID//'.dat',unit=91+TID, status='old')
    allocate(L_curve(K-2,4))
    do i=1,K-2
      read(91+TID,'(E15.8,X,E15.8,X,E15.8)')L_curve(i,1),L_curve(i,3),L_curve(i,2)
    enddo
    svar=minval(L_curve(:,2))
    L_curve(:,2)=L_curve(:,2)-svar
    svar=minval(L_curve(:,3))
    L_curve(:,3)=L_curve(:,3)-svar
    L_curve(:,4)=sqrt(l_curve(:,2)**2+l_curve(:,3)**2)
    i=minloc(L_curve(:,4),dim=1)
    tol_save_graphical=l_curve(i,1)
    close(91+TID)
  else
    print*, 'NO VALID PSEUDOINVERSE OPTION GIVEN'
    stop
  endif
  if(check_vec_zero)x_0=0.d0
  deallocate(u)
  deallocate(s)
  deallocate(vt)

end subroutine tolerance_simple

subroutine Pseudoinverse(m,n,a,pinv_a,pinv_tol,residue_opt)
  implicit none
  !M is usually 3*nat, N is usually IA_DOF
  integer n,m,i,K
  character*1 residue_opt!T=TIKHONOV, M=MOORE-PENROSE
  real(8) a(m,n),pinv_a(n,m),a_in(m,n),pinv_tol
  real(8),allocatable :: u(:,:),s(:),vt(:,:),s_store(:,:)

  a_in=a
  K = MIN(M,N)

  allocate(u(m,K))
  allocate(s(K))
  allocate(vt(K,n))

  call svd(m,n,K,a,s,u,vt)
  allocate(s_store(k,k))
  s_store=0.d0
  if(residue_opt.eq.'T')then
    do i=1,k
      s_store(i,i)=s(i)/(s(i)**2+pinv_tol**2)
    enddo
  elseif(residue_opt.eq.'M')then
    do i=1,k
      if(dabs(s(i)).gt.pinv_tol)then
        s_store(i,i)=1.d0/s(i)
      else
        s_store(i,i)=1.d0/(maxval(abs(s))*dble(k))
      endif
    enddo
  endif
  deallocate(s)

  pinv_a=matmul(transpose(vt),matmul(s_store,transpose(u)))
  deallocate(s_store)
  deallocate(u)
  deallocate(vt)

  a=a_in

  return
end subroutine Pseudoinverse

subroutine input_preprocessing(ni,ns,ntest,xs,z_pca,major_comp_keep,eigvectr,&
remove_n_dims,mu,variance)
  implicit none
  integer, intent(in)   :: ni,ns,ntest,remove_n_dims
  real(8), intent(inout):: xs(ni,ns+ntest)
  real(8), intent(inout):: z_pca(ni,ns+ntest)
  integer, intent(inout):: major_comp_keep
  real(8), intent(inout):: eigvectr(ni,ni)
  real(8), intent(inout):: mu(ni)
  real(8), intent(inout):: variance(ni)
  real(8) xs_scaled(ni,ns+ntest)!,rp,trace
  real(8), dimension(:,:), allocatable :: covariance,&
  z_pca_store,eigvectr_store
  real(8), dimension(:), allocatable   :: eignum_r
  integer i,j

  mu=0.d0
!  do j=1,ns
!    mu(:)=mu(:)+xs(:,j)
!  enddo
!  mu=mu/dble(ns)
!  if(coords_inverse)then
!    mu=0.d0
!    do j=1,ns
!      mu(:)=mu(:)+1.d0/xs(:,j)
!    enddo
!    mu=mu/dble(ns)
!    mu=1.d0/mu
!  endif

!!  if(tolerance.lt.1.d0 .and. .not.interatomic_coords)then

  do i=1,ns+ntest
    xs_scaled(:,i)=xs(:,i)-mu(:)
  enddo

  variance=0.d0
  do i=1,ni
    do j=1,ns
      variance(i)=variance(i)+xs_scaled(i,j)**2
    enddo
  enddo
  variance=variance/dble(ns)
  variance=sqrt(variance)
! I DON'T THINK RESCALING WRT VARIANCE IS A GREAT IDEA WITH MASS-WEIGHTED COORDS
!!  if(interatomic_coords .and. coords_inverse)then
!  do i=1,ni
!    xs_scaled(i,:)=xs_scaled(i,:)/variance(i)
!  enddo
  xs=xs_scaled
!!  endif
  allocate(covariance(ni,ni))

  covariance=matmul(xs_scaled(:,1:ns),transpose(xs_scaled(:,1:ns)))/dble(ns)

  allocate(eignum_r(ni))
  call r_diagonal(ni,covariance,eignum_r,eigvectr)
  allocate(eigvectr_store(ni,ni))
  j=0
  do i=ni,1,-1
    j=j+1
    print*,'PCA eigval',j,eignum_r(i)
    eigvectr_store(:,j)=eigvectr(:,i)
  enddo
  eigvectr=eigvectr_store
  deallocate(eigvectr_store)
  deallocate(covariance)
!  rp=0.d0
!  trace=sum(dsqrt(dabs(eignum_r)**2+dabs(eignum_i)**2),dim=1)
! TOLERANCE CRITERIA FOR DIMENSION REDUCTION REMOVED, USER SIMPLY DEFINES HOW MANY DIMS SHOULD BE REMOVED.
!  if(tolerance.lt.1.d0)then
!    do while(rp.lt.tolerance)
!      major_comp_keep=major_comp_keep+1
!      rp=sum(dsqrt(dabs(eignum_r(1:major_comp_keep))**2+dabs(eignum_i(1:major_comp_keep))**2),dim=1)/trace
!    enddo
!  else
!    major_comp_keep=ni
!  endif

  major_comp_keep=ni-remove_n_dims
  deallocate(eignum_r)
  z_pca=0.d0
  allocate(z_pca_store(major_comp_keep,ns+ntest))
  z_pca_store=matmul(transpose(eigvectr(:,1:major_comp_keep)),xs_scaled)
  z_pca(1:major_comp_keep,:)=z_pca_store
  deallocate(z_pca_store)
  print*,'MAJOR COMPONENTS RETAINED',major_comp_keep,'. THERE ARE',ni-major_comp_keep,'FEWER DIMENSIONS.'

end subroutine input_preprocessing

subroutine print_coords(ncoord_internal,ncoord_cart,npoint,refcoords,xcoords,refene)
  implicit none
  integer,intent(in) :: ncoord_internal,ncoord_cart,npoint
  real(8),intent(in) :: refcoords(ncoord_internal,npoint)
  real(8),intent(in) :: xcoords(ncoord_cart,npoint)
  real(8),intent(in) :: refene(npoint)
  integer :: icoord,ipoint,ndist,iat,jat,nat,idist
  real(8) :: sortlist(npoint),intervall
  real(8),allocatable :: dist(:,:)

  intervall=(maxval(refene)-minval(refene))*0.5D0

  ! normal coordinates
  open(unit=103,file='trainingsset.normal_coords')
  write(103,*) "# Energy of control points of training set vs. normal coordinates"
  write(103,*) "# The energy is shifted for each coordinate for better visibility"
  do icoord=1,ncoord_internal
    write(103,'("#Coordinate ",i3," energies - ",f10.5)') &
        icoord,dble(icoord-1)*intervall
    sortlist=refcoords(icoord,:)
    do ipoint=1,npoint
      write(103,*) minval(sortlist),refene(minloc(sortlist,dim=1))-dble(icoord-1)*intervall
      sortlist(minloc(sortlist,dim=1))=huge(1.D0)
    end do
    write(103,*)
  end do
  close(103)

  ! cartesian coordinates
  open(unit=103,file='trainingsset.cartesian_coords')
  write(103,*) "# Energy of control points of training set vs. cartesian coordinates"
  write(103,*) "# The energy is shifted for each coordinate for better visibility"
  do icoord=1,ncoord_cart
    write(103,'("#Coordinate ",i3," energies - ",f10.5)') &
        icoord,dble(icoord-1)*intervall
    sortlist=xcoords(icoord,:)
    do ipoint=1,npoint
      write(103,*) minval(sortlist),refene(minloc(sortlist,dim=1))-dble(icoord-1)*intervall
      sortlist(minloc(sortlist,dim=1))=huge(1.D0)
    end do
    write(103,*)
  end do
  close(103)

  ! interatomic distances
  nat=ncoord_cart/3
  ndist=nat*(nat-1)/2
  allocate(dist(ndist,npoint))
  do ipoint=1,npoint
    idist=1
    do iat=1,nat
      do jat=iat+1,nat
        if(idist>ndist) stop "wrong number of distances"
        dist(idist,ipoint)=sqrt(sum((xcoords(3*iat-2:3*iat,ipoint)-xcoords(3*jat-2:3*jat,ipoint))**2))
        idist=idist+1
      end do
    end do
  end do
  open(unit=103,file='trainingsset.distances')
  write(103,*) "# Energy of control points of training set vs. interatomic dinstances"
  write(103,*) "# The energy is shifted for each coordinate for better visibility"
  do icoord=1,ndist
    write(103,'("#Coordinate ",i3," energies - ",f10.5)') &
        icoord,dble(icoord-1)*intervall
    sortlist=dist(icoord,:)
    do ipoint=1,npoint
      write(103,*) minval(sortlist),refene(minloc(sortlist,dim=1))-dble(icoord-1)*intervall
      sortlist(minloc(sortlist,dim=1))=huge(1.D0)
    end do
    write(103,*)
  end do
  close(103)
  open(unit=103,file='trainingsset.invdistances')
  write(103,*) "# Energy of control points of training set vs. inverse interatomic dinstances"
  write(103,*) "# The energy is shifted for each coordinate for better visibility"
  do icoord=1,ndist
    write(103,'("#Coordinate ",i3," energies - ",f10.5)') &
        icoord,dble(icoord-1)*intervall
    sortlist=dist(icoord,:)
    do ipoint=1,npoint
      write(103,*) 1.D0/minval(sortlist),refene(minloc(sortlist,dim=1))-dble(icoord-1)*intervall
      sortlist(minloc(sortlist,dim=1))=huge(1.D0)
    end do
    write(103,*)
  end do
  close(103)

end subroutine print_coords

real(8) function kronecker(i,j)
  implicit none
  integer, intent(in) :: i,j

  if(i.eq.j)then
    kronecker=1.d0
  else
    kronecker=0.d0
  endif
  return

end function kronecker

subroutine SVD(m,n,K,a,S,U,VT)
  implicit integer(i-n)
  implicit real(8)(a-h,o-z)
  integer n,m,info,lwork,K,i
  real(8) a(m,n),S(min(m,n)),U(m,K),VT(K,n)
  character*1 JOBU,JOBVT
  real(8), dimension(:),allocatable:: WORK

  allocate(WORK(1))
  lwork=-1
  JOBU='S'
  JOBVT='S'

  CALL dgesvd( JOBU, JOBVT, M, N, A, M, S, U, M,&
  vt, K, work, lwork, info )

  lwork=nint(WORK(1))
  deallocate(work)
  allocate(work(lwork))

  CALL dgesvd( JOBU, JOBVT, M, N, A, M, S, U, M,&
  vt, K, work, lwork, info )

  deallocate(work)

  do i=1,min(m,n)
    if(abs(S(i)).lt.1.d-16)then
      S(i)=1.d-16
    endif
  enddo

  return
end subroutine SVD

subroutine SVD_A(m,n,a,S,U,VT)
  implicit integer(i-n)
  implicit real(8)(a-h,o-z)
  integer n,m,info,lwork,i
  real(8) a(m,n),S(min(m,n)),U(m,m),VT(n,n)
  character*1 JOBU,JOBVT
  real(8), dimension(:),allocatable:: WORK

  allocate(WORK(1))
  lwork=-1
  JOBU='A'
  JOBVT='A'

  CALL dgesvd( JOBU, JOBVT, M, N, A, M, S, U, M,&
  vt, N, work, lwork, info )

  lwork=nint(WORK(1))
  deallocate(work)
  allocate(work(lwork))

  CALL dgesvd( JOBU, JOBVT, M, N, A, M, S, U, M,&
  vt, N, work, lwork, info )

  deallocate(work)

  do i=1,min(m,n)
    if(abs(S(i)).lt.1.d-16)then
      S(i)=1.d-16
    endif
  enddo

  return
end subroutine SVD_A

character(2) function get_atom_symbol(atomic_number)
  implicit none
  integer, intent(in) :: atomic_number
  character(2), parameter :: elements(111) = &
       (/ 'H ','He', &
          'Li','Be','B ','C ','N ','O ','F ','Ne', &
          'Na','Mg','Al','Si','P ','S ','Cl','Ar', &
          'K ','Ca','Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu', &
          'Zn','Ga','Ge','As','Se','Br','Kr', &
          'Rb','Sr','Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag', &
          'Cd','In','Sn','Sb','Te','I ','Xe', &
          'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy', &
          'Ho','Er','Tm','Yb','Lu','Hf','Ta','W ','Re','Os','Ir','Pt', &
          'Au','Hg','Tl','Pb','Bi','Po','At','Rn', &
          'Fr','Ra','Ac','Th','Pa','U ','Np','Pu','Am','Cm','Bk','Cf', &
          'Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds', &
          'Rg' /)
! **********************************************************************
  if (atomic_number >= 1 .and. atomic_number <= size(elements)) then
     get_atom_symbol = elements(atomic_number)
  else
     get_atom_symbol = 'XX'
  endif
end function get_atom_symbol

subroutine vary_vector(x_in,nat,delta_x,delta_r,i_red)
  implicit none
  integer, intent(in) :: nat,i_red
  real(8), intent(in) :: x_in(3*nat)
  real(8), intent(out):: delta_x(3*nat),delta_r(i_red)
  real(8) rand_val,radii(i_red),x_var(3*nat)
  integer i,j,k

  k=0
  do i=1,nat
    do j=i+1,nat
      k=k+1
      radii(k)=sqrt(sum((x_in(3*i-2:3*i)-x_in(3*j-2:3*j))**2))
    enddo
  enddo
  
  do i=1,3*nat
    call random_seed()
    call random_number(rand_val)
    rand_val=2.d0*rand_val-1.d0
    rand_val=rand_val/1.d3
    delta_x(i)=x_in(i)*rand_val
    x_var(i)=x_in(i)+x_in(i)*rand_val
  enddo
  k=0
  do i=1,nat
    do j=i+1,nat
      k=k+1
      delta_r(k)=sqrt(sum((x_var(3*i-2:3*i)-x_var(3*j-2:3*j))**2))-radii(k)
    enddo
  enddo

end subroutine vary_vector

RECURSIVE FUNCTION Factorial(n)  RESULT(Fact)

  IMPLICIT NONE
  INTEGER :: Fact
  INTEGER, INTENT(IN) :: n

  IF (n == 0) THEN
     Fact = 1
  ELSE
     Fact = n * Factorial(n-1)
  END IF

END FUNCTION Factorial

REAL(8) FUNCTION NCR(n,r)
  IMPLICIT NONE
  INTEGER, intent(in) :: N,R
  INTEGER I,J
!  INTEGER(8) NCRSTORE
  REAL(8) NCRSTORE_REAL
  J=N-R
!  NCRSTORE=1
!  DO I=J+1,N
!    NCRSTORE=NCRSTORE*I
!  ENDDO
!  DO I=1,R
!    NCRSTORE=NCRSTORE/I
!  ENDDO
!  NCR=NCRSTORE
  NCRSTORE_REAL=0.d0
  DO I=J+1,N
    NCRSTORE_REAL=NCRSTORE_REAL+LOG(DBLE(I))
  ENDDO
  DO I=1,R
    NCRSTORE_REAL=NCRSTORE_REAL-LOG(DBLE(I))
  ENDDO
  NCR=NCRSTORE_REAL

END FUNCTION NCR

subroutine nexksb(n,k,a,mtc,h,m2)
  IMPLICIT NONE
  integer, intent(in) :: n,k
  integer, intent(inout) :: a(k),h,m2
  integer j
  logical, intent(inout) :: mtc
  j=0
  if(k.le.0)then
    a=0
    mtc=.false.
    return
  else
    if(.not.mtc)then
      m2=0
      h=k
    else
      if(m2.lt.n-h) h=0
      h=h+1
      m2=a(k+1-h)
    endif
  endif
  do j=1,h
    a(k+j-h)=m2+j
  enddo
  mtc=a(1).ne.n-k+1
  return
end subroutine nexksb

subroutine Shuffle(a,n)
  integer, intent(in) :: n
  integer, intent(inout) :: a(n)
  integer :: i, randpos, temp
  real :: r

  do i = n, 2, -1
    call random_number(r)
    randpos = int(r * i) + 1
    temp = a(randpos)
    a(randpos) = a(i)
    a(i) = temp
  end do
 
end subroutine Shuffle

subroutine DMAT_NR_singularity_check(trans_coords,omit_bonds,i_rad_rem,nat,i_red,ns,ntest,x_outr,x_out,x_unchanged,&
  g_unchanged,h_unchanged,smallest_singular,d_or_v_check,check_type,pinv_tol,coords_inverse)
  USE OMP_LIB
  implicit none
  integer, intent(in) :: ns,ntest,nat,i_red,i_rad_rem,omit_bonds(i_rad_rem)
  real(8), intent(in) :: x_unchanged(3*nat,ns+ntest),g_unchanged(3*nat,ns+ntest),h_unchanged(3*nat,3*nat,ns+ntest),&
  x_outr(i_red,ns+ntest),trans_coords(i_red-i_rad_rem,i_red-i_rad_rem),pinv_tol
  logical, intent(in) :: coords_inverse
  character(1), intent(in):: check_type
  real(8), intent(out):: smallest_singular,x_out(i_red-i_rad_rem,ns+ntest),d_or_v_check
  real(8), allocatable :: DMAT(:,:,:),&!ncoord,3*nat,ns+ntest)
  S(:),U(:,:),VT(:,:),delta_x(:,:),delta_r(:,:),delta_r_nr(:,:),PINV(:,:),&
  projection(:,:,:),g_r(:),PINV2(:,:),h_r(:,:)
  integer i,j,k,m,l,counter,ncoord
  real(8) dist(2),rp
  character(2) prj_len

!  allocate(x_rmw(i_red-i_rad_rem,ns+ntest))

  ncoord=i_red-i_rad_rem
  counter=0
  k=0
  do i=1,nat
    do j=i+1,nat
      counter=counter+1
      if(.not.any(omit_bonds==counter))then
        k=k+1
        x_out(k,:)=x_outr(counter,:)
!        x_rmw(k,:)=x_outr_mw(counter,:)
      endif
    enddo
  enddo
!  do k=1,ns+ntest
!    x_rmw(:,k)=matmul(trans_coords,x_rmw(:,k))
!  enddo

  smallest_singular=0.d0
  d_or_v_check=0.d0
  select case(check_type)
    case('S')

      allocate(DMAT(ncoord,3*nat,ns+ntest))
      call make_DMAT(nat,i_red,ncoord,ns,ntest,DMAT,x_out,x_unchanged,&
      omit_bonds,coords_inverse)
      do k=1,ns+ntest
        DMAT(:,:,k)=matmul(trans_coords,DMAT(:,:,k))
      enddo
      allocate(S(min(3*nat,ncoord)))
      allocate(VT(ncoord,ncoord))
      allocate(U(3*nat,3*nat))
      !$OMP PARALLEL DO PRIVATE(k,S,U,VT) SHARED(ns,ntest,nat,ncoord,DMAT,smallest_singular),&
      !$OMP default(none)
      do k=1,ns+ntest
        call SVD_A(3*nat,ncoord,transpose(DMAT(:,:,k)),S,U,VT)
        smallest_singular=smallest_singular+minval(abs(S(1:min(3*nat-6,ncoord))))/maxval(abs(S))!sum(log(abs(S)/S(1)),dim=1)
      enddo
      !$OMP END PARALLEL DO
      smallest_singular=smallest_singular/dble(ns+ntest)
      deallocate(S)
      deallocate(U)
      deallocate(VT)
      deallocate(DMAT)

    case('D')

!      allocate(DMAT(ncoord,3*nat,ns+ntest))
!      allocate(barycentre(3,ns+ntest))
!      barycentre=0.d0
!      call make_DMAT(nat,i_red,ncoord,ns,ntest,DMAT,x_out,x_unchanged,barycentre,&
!      refmass,omit_bonds,coords_inverse,.false.)
!      deallocate(barycentre)

!      allocate(S(min(3*nat,ncoord)))
!      allocate(VT(ncoord,ncoord))
!      allocate(U(3*nat,3*nat))
!      smallest_singular=0.d0

      !$OMP PARALLEL DO PRIVATE(k,m)&
      !$OMP SHARED(i_red,i_rad_rem,coords_inverse,d_or_v_check,x_out,ns,ntest,nat,ncoord),&
      !$OMP default(none)
      do k=1,ns+ntest
!        call SVD_A(3*nat,ncoord,transpose(DMAT(:,:,k)),S,U,VT)
!        smallest_singular=smallest_singular+sum(log(abs(S)/S(1)),dim=1)
        do m=1,i_red-i_rad_rem
          if(.not.coords_inverse)then
            d_or_v_check=d_or_v_check+x_out(m,k)/maxval(x_out(:,k))
          else            
            d_or_v_check=d_or_v_check+(1.d0/x_out(m,k))/(1.d0/minval(x_out(:,k)))
          endif
        enddo
      enddo
      !$OMP END PARALLEL DO
!      d_or_v_check=d_or_v_check/smallest_singular
      d_or_v_check=d_or_v_check/dble(ns+ntest)
!      deallocate(S)
!      deallocate(U)
!      deallocate(VT)
!      deallocate(DMAT)

    case('V')

      allocate(delta_x(3*nat,ns+ntest))
      allocate(delta_r(i_red,ns+ntest))
      do i=1,ns+ntest
        call vary_vector(x_unchanged(:,i),nat,delta_x(:,i),delta_r(:,i),i_red)
        delta_r(:,i)=matmul(trans_coords,delta_r(:,i))
!        dist=huge(1.d0)
!        do j=1,ns+ntest
!          if(j.ne.i)then
!            call coord_abstand(3*nat,x_unchanged(:,i),x_unchanged(:,j),dist(2))
!            if(dist(2).ne.0.d0)then
!              if(dist(2).lt.dist(1))then
!                dist(1)=dist(2)
!                delta_x(:,i)=x_unchanged(:,i)-x_unchanged(:,j)
!                delta_r(:,i)=x_outr(:,i)-x_outr(:,j)
!              endif
!            endif
!          endif
!        enddo
      enddo

      allocate(delta_r_nr(ncoord,1))
!      allocate(S(min(3*nat,ncoord)))
!      allocate(VT(3*nat,3*nat))
!      allocate(U(ncoord,ncoord))
!      allocate(SI(ncoord,3*nat))

      allocate(DMAT(ncoord,3*nat,ns+ntest))
      call make_DMAT(nat,i_red,ncoord,ns,ntest,DMAT,x_out,x_unchanged,&
      omit_bonds,coords_inverse)
      do k=1,ns+ntest
        DMAT(:,:,k)=matmul(trans_coords,DMAT(:,:,k))
      enddo
      allocate(PINV(3*nat,ncoord))
      allocate(projection(ncoord,ncoord,2))
      write(prj_len,'(I2.1)')ncoord
      do k=1,ns

        call Pseudoinverse(ncoord,3*nat,DMAT(:,:,k),PINV,pinv_tol,'M')
        projection(:,:,1)=matmul(DMAT(:,:,k),PINV)
!        call Pseudoinverse(ncoord,ncoord,projection(:,:,1),projection(:,:,2),pinv_tol,'M')
        projection(:,:,2)=transpose(projection(:,:,1))
!        do l=1,ncoord
!          write(*,'(I2.1,'//prj_len//'E12.5)') l,projection(l,:,1)
!        enddo
!        pause

!        call SVD_A(ncoord,3*nat,DMAT(:,:,k),S,U,VT)
!        do m=1,min(ncoord,3*nat)
!          if(S(m).gt.pinv_tol)then
!            SI(m,m)=1.d0/S(m)
!          else
!            SI(m,m)=0.d0
!          endif
!        enddo
        l=0
        do m=1,i_red
          if(.not.any(omit_bonds==m))then
            l=l+1
            delta_r_nr(l,1)=delta_r(m,k)
          endif
        enddo
        d_or_v_check=d_or_v_check+log(sum((matmul(PINV,matmul(projection(:,:,2),delta_r_nr(:,1)))-&
        delta_x(:,k))**2,dim=1))+log(sum((matmul(DMAT(:,:,k),delta_x(:,k))-&
        matmul(projection(:,:,2),delta_r_nr(:,1)))**2,dim=1))
      enddo
      d_or_v_check=d_or_v_check/dble(ns)
      deallocate(projection)
      deallocate(PINV)
      deallocate(DMAT)
      deallocate(delta_r)
      deallocate(delta_x)
      deallocate(delta_r_nr)
!      deallocate(S)
!      deallocate(U)
!      deallocate(VT)
!      deallocate(SI)
    case('G')
      allocate(DMAT(ncoord,3*nat,ns+ntest))
      call make_DMAT(nat,i_red,ncoord,ns,ntest,DMAT,x_out,x_unchanged,&
      omit_bonds,coords_inverse)
      do k=1,ns+ntest
        DMAT(:,:,k)=matmul(trans_coords,DMAT(:,:,k))
      enddo
      allocate(PINV(ncoord,3*nat))
      allocate(g_r(ncoord))
      !$OMP PARALLEL DO PRIVATE(k,g_r,pinv_tol) SHARED(ns,ntest,nat,ncoord,DMAT,PINV,g_unchanged,rp,&
      !$OMP d_or_v_check),&
      !$OMP default(none)
      do k=1,ns+ntest
        call Pseudoinverse(3*nat,ncoord,transpose(DMAT(:,:,k)),PINV,pinv_tol,'M')
        g_r=matmul(PINV,g_unchanged(:,k))
        rp=sum(g_r,dim=1)/dble(ncoord)
        d_or_v_check=d_or_v_check+log(sum((g_r-matmul(DMAT(:,:,k),g_unchanged(:,k)))**2,dim=1))&
        +log(sum((g_r-rp)**2)&
        )
      enddo
      !$OMP END PARALLEL DO
      d_or_v_check=d_or_v_check/dble(ns+ntest)
      deallocate(PINV)
      deallocate(DMAT)
      deallocate(g_r)
    case('H')
      allocate(DMAT(ncoord,3*nat,ns+ntest))
      call make_DMAT(nat,i_red,ncoord,ns,ntest,DMAT,x_out,x_unchanged,&
      omit_bonds,coords_inverse)
      do k=1,ns+ntest
        DMAT(:,:,k)=matmul(trans_coords,DMAT(:,:,k))
      enddo
      allocate(PINV(3*nat,ncoord))
      allocate(PINV2(ncoord,3*nat))
      allocate(h_r(ncoord,ncoord))
      allocate(U(ncoord,ncoord))
      !$OMP PARALLEL DO PRIVATE(k,pinv_tol,h_r,rp,U,dist) SHARED(nat,ncoord,DMAT,PINV2,h_unchanged,pinv,d_or_v_check,ns,ntest),&
      !$OMP default(none)
      do k=1,ns+ntest
        call Pseudoinverse(3*nat,ncoord,transpose(DMAT(:,:,k)),PINV2,pinv_tol,'M')
        call Pseudoinverse(ncoord,3*nat,DMAT(:,:,k),PINV,pinv_tol,'M')
        h_r=matmul(pinv2,matmul(h_unchanged(:,:,k),pinv))
        rp=0.d0
        do i=1,ncoord
          do j=1,ncoord
            if(abs(h_r(i,j)).lt.1.d8)then
              rp=rp+h_r(i,j)
            endif
          enddo
        enddo
        rp=rp/dble(ncoord)**2
        U=matmul(transpose(DMAT(:,:,k)),matmul(h_r,DMAT(:,:,k)))
        dist=0.d0
        do i=1,3*nat
          do j=1,3*nat
            dist(1)=dist(1)+(U(j,i)-h_unchanged(j,i,k))**2
            if((h_r(j,i)-rp)**2.lt.1.d15)then
              dist(2)=dist(2)+(h_r(j,i)-rp)**2
            endif
          enddo
        enddo
        d_or_v_check=d_or_v_check+log(dist(1))!+log(dist(2))
      enddo
      !$OMP END PARALLEL DO
      d_or_v_check=d_or_v_check/dble(ns+ntest)
      deallocate(PINV)
      deallocate(PINV2)
      deallocate(DMAT)
      deallocate(h_r)
      deallocate(U)
    case default
      print*, 'NO VALID CHECK TYPE GIVEN'
      stop
  end select

end subroutine DMAT_NR_singularity_check

subroutine non_redundant_smallest_distances(nat,ncoord,i_red,ns,ntest,x_outr,x_out,omit_bonds,coords_inverse)
  implicit none
  integer, intent(in) :: nat,ncoord,i_red,ns,ntest
  real(8), intent(in) :: x_outr(i_red,ns+ntest)
  logical, intent(in) :: coords_inverse
  real(8), intent(out):: x_out(ncoord,ns+ntest)
  integer, intent(out):: omit_bonds(i_red-ncoord)
  integer, allocatable :: mapping(:,:), back_mapping(:,:),removable_bonds(:)
  real(8), allocatable :: x_r(:),frequency(:,:),inst_freq(:)
  integer i,red_num,j,k,iat,jat,l,counter
  real(8) penalty
  logical permitted_atom(nat),permitted_bond(i_red)

  red_num=i_red-ncoord

  allocate(back_mapping(i_red,2))
  back_mapping=0
  allocate(mapping(nat,nat))
  mapping=0
  counter=0
  do i=1,nat
    do j=i+1,nat
      counter=counter+1
      back_mapping(counter,1)=i
      back_mapping(counter,2)=j
      mapping(i,j)=counter
      mapping(j,i)=counter
    enddo
  enddo

  !FIND THE LOCATION OF THE LARGEST DISTANCES FOR EACH CONFIG IN TRAINING SET
  allocate(x_r(i_red))
  allocate(frequency(i_red,red_num))
  allocate(inst_freq(nat))
  allocate(removable_bonds(nat))
  frequency=0.d0
  do i=1,ns+ntest
    l=0
    x_r(:)=x_outr(:,i)!x_outr_mw(:,i)
    inst_freq=0.d0
    permitted_atom=.true.
    removable_bonds=nat-3
    l=0
    do j=1,i_red
      if(l.eq.red_num)exit
      if(coords_inverse)then
        k=minloc(x_r,dim=1)
        penalty=huge(1.d0)
      else
        penalty=-1.d0
        k=maxloc(x_r,dim=1)
      endif
!      penalty=-1.d0

      iat=back_mapping(k,1)
      jat=back_mapping(k,2)
      if(nint(inst_freq(iat)).le.removable_bonds(iat)&
      .and. nint(inst_freq(jat)).le. removable_bonds(jat))then
        if(permitted_atom(iat) .and. permitted_atom(jat))then
          l=l+1
          if(coords_inverse)then
            frequency(k,l)=frequency(k,l)+1.d0/x_r(k)
          else
            frequency(k,l)=frequency(k,l)+x_r(k)
          endif

          inst_freq(iat)=inst_freq(iat)+1.d0
          inst_freq(jat)=inst_freq(jat)+1.d0
          x_r(k)=penalty
        else
          x_r(k)=penalty
        endif
      elseif(nint(inst_freq(iat)).gt. removable_bonds(iat))then
        permitted_atom(iat)=.false.
        inst_freq(iat)=dble(3*nat)
        removable_bonds(jat)=removable_bonds(jat)-1
        x_r(k)=penalty
      elseif(nint(inst_freq(jat)).gt. removable_bonds(jat))then
        permitted_atom(jat)=.false.
        inst_freq(jat)=dble(3*nat)
        removable_bonds(iat)=removable_bonds(iat)-1
        x_r(k)=penalty
      endif
    enddo
  enddo
  deallocate(x_r)
  deallocate(inst_freq)
  allocate(inst_freq(i_red))
  inst_freq=0.d0
  do i=1,red_num
    inst_freq=inst_freq+frequency(:,i)
  enddo
  deallocate(frequency)

  permitted_bond=.true.
  removable_bonds=nat-3
  j=0
  do i=1,i_red
    if(j.eq.red_num)exit
    k=maxloc(inst_freq,dim=1,mask=inst_freq.gt.0)
    if(i.eq.1)then
      j=j+1
      omit_bonds(j)=k
      removable_bonds(back_mapping(k,1))=removable_bonds(back_mapping(k,1))-1
      removable_bonds(back_mapping(k,2))=removable_bonds(back_mapping(k,2))-1
      permitted_bond(k)=.false.
    elseif(i.gt.1)then
      if(removable_bonds(back_mapping(k,1)).gt.0.and.removable_bonds(back_mapping(k,2)).gt.0 .and.&
      permitted_bond(k))then
        j=j+1
        omit_bonds(j)=k
        removable_bonds(back_mapping(k,1))=removable_bonds(back_mapping(k,1))-1
        removable_bonds(back_mapping(k,2))=removable_bonds(back_mapping(k,2))-1
        permitted_bond(k)=.false.
      endif
    endif
    inst_freq(k)=-1.d0
  enddo
!  print*,omit_bonds
  deallocate(removable_bonds)
  deallocate(inst_freq)
  j=0
  do i=1,i_red
    if(.not.any( omit_bonds==i ))then
      j=j+1
      x_out(j,:)=x_outr(i,:)
    endif
  enddo

end subroutine non_redundant_smallest_distances
