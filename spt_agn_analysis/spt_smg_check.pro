FUNCTION return_wendy_fluxes, band

  if band eq '90' then begin 
;  90GHz sources: 
     ra = [ 338.21308333, 347.849875 , 349.84016667, 353.11041667, 353.80816667, 355.03733333, 357.42583333, 357.37258333, 357.42825  ]
     dec = [-61.24577778, -54.84169444, -55.96605556, -53.97772222, -53.40583333, -59.72563889, -50.89191667, -52.78016667, -56.63977778]

     wendy_flux = [ 2.53276 , 2.07869 , 2.29415 , -8.29094 , 2.23367 , 0.357033, 0.436138, 0.09987 , 0.574826]
     wendy_ferr = [ 0.0376  , 0.041924 , 0.0880625, 9.157539 , 0.1772425, 1.097229 , 1.0103995, 1.3139835, 1.165794 ]

     if 0 then begin ;these are tom's from the cluster quick run
        ps_master = '/data/bleeml/SPT_Planck_combined_maps/point_source_files/ps_master.sav'
        restore, ps_master, /v  ;ps
        wendy_Flux = fltarr(n_elements(ra))
        wendy_ferr = wendy_flux
        for i=0, n_elements(ra)-1, 1 do begin
           gcirc2, 2, ra[i], dec[i], ps.ra, ps.dec, dis
           q = where(dis/60.0 le 1)
           if q[0] ne -1 then begin
              wendy_flux[i] = ps[q[0]].f90
              wendy_ferr[i] = wendy_flux[i]/ps[q[0]].sn90
           endif
        endfor
     endif
  endif

  if band eq '150' then begin 
;150GHz sources:
     ra = [ 347.849875 , 349.84016667, 353.11041667, 353.80816667, 355.03733333, 357.42583333, 357.37258333, 357.42825  ]
     dec = [-54.84169444, -55.96605556, -53.97772222, -53.40583333, -59.72563889, -50.89191667, -52.78016667, -56.63977778]
     wendy_flux = [ 5.43231 , 5.08956 , -0.811166, 4.58064 , 4.76949 , 3.85418 , 4.22777 , 6.01398 ]
     wendy_ferr = [ 0.201465, 0.173135, 5.267946, 0.70147 , 0.29484 , 0.293465, 0.26898 , 0.262365]
     if 0 then begin 
        ps_master = '/data/bleeml/SPT_Planck_combined_maps/point_source_files/ps_master.sav'
        restore, ps_master, /v  ;ps
        wendy_Flux = fltarr(n_elements(ra))
        wendy_ferr = wendy_flux
        for i=0, n_elements(ra)-1, 1 do begin
           gcirc2, 2, ra[i], dec[i], ps.ra, ps.dec, dis
           q = where(dis/60.0 le 1)
           if q[0] ne -1 then begin
              wendy_flux[i] = ps[q[0]].f150
              wendy_ferr[i] = wendy_flux[i]/ps[q[0]].sn150
           endif
        endfor
     endif
  endif

  if band eq '220' then begin 
;220GHz sources:
     ra = [ 347.849875 , 349.84016667, 353.11041667, 353.80816667, 355.03733333, 357.42583333, 357.37258333, 357.42825  ]
     dec =  [-54.84169444, -55.96605556, -53.97772222, -53.40583333, -59.72563889, -50.89191667, -52.78016667, -56.63977778] 
     wendy_flux = [ 18.8499, 16.8083, 33.0132, 12.4632, 14.9155, 14.8363, 15.0811, 20.1811]
     wendy_ferr = [ 1.39015 , 1.68085 , 7.5105 , 0.872415, 2.33057 , 4.71055 , 3.713715, 1.10995 ]

     if 0 then begin 
     ;temporary since that array isn't right. 
        ps_master = '/data/bleeml/SPT_Planck_combined_maps/point_source_files/ps_master.sav'
        restore, ps_master, /v  ;ps
        wendy_Flux = fltarr(n_elements(ra))
        wendy_ferr = wendy_flux
        for i=0, n_elements(ra)-1, 1 do begin
           gcirc2, 2, ra[i], dec[i], ps.ra, ps.dec, dis
           q = where(dis/60.0 le 1)
           if q[0] ne -1 then begin
              wendy_flux[i] = ps[q[0]].f220
              wendy_ferr[i] = wendy_flux[i]/ps[q[0]].sn220
           endif
        endfor
     endif
  endif
  temp  = {ra:0.0d, dec:0.0d, flux:0.0d, flux_err:0.0d, old_sn:0.0, new_sn:0.0, new_uk:0.0, new_flux:0.0, ratio:0.0, off_x:0.0, off_y:0.0}
  out = replicate(temp, n_elements(ra))
  out.ra = ra
  out.dec = dec
  out.flux = wendy_flux
  out.flux_err = wendy_ferr
  out.old_sn = wendy_flux/wendy_ferr
  return, out
end




FUNCTION pull_out_smg_fluxes

  field = 'ra23h30dec-55_2year'
  band = ['90', '150', '220']
  outdir = '/data/bleeml/for_amy/smg_checks/'

  for i=0, n_elements(band)-1, 1 do begin 
     outfile = outdir + field + '_'+band[i] + '_check.sav'
     restore, outfile
     ngrid_x = n_elements(szmaps_filt[*,0])
     ngrid_y = n_elements(szmaps_filt[0,*])
     
     ws = return_wendy_fluxes(band[i])
     ang2pix_proj0, ws.ra, ws.dec, [ngrid_x, ngrid_y], radec0, reso_arcmin, iptemp, xp=xps, yp=yps
     ws.new_sn = szmaps_filt[xps, yps]
     ws.new_uk = filtered_map[xps, yps]

     ws.off_x = (xps-ngrid_x/2.0)/(ngrid_x/2.0)
     ws.off_y = (yps-ngrid_y/2.0)/(ngrid_y/2.0)

     convert = mjy_to_uk(1e6,freq=band[i],omeg=computearea,/inv)
     ;convert2 = mjy_to_uk(1e6, freq=band[i], omega=solid_angle, /inv)
     
     ws.new_flux = ws.new_uk*convert
     ws.ratio = ws.new_flux/ws.flux
     print_struct, ws
     stop
  endfor

  return,0
END
     
       
pro spt_smg_check

  ;do a quick extraction check on SMG fluxes from the 23h field 

  field = 'ra23h30dec-55_2year'
  band = ['90', '150', '220']
  proj=0
  
  apmask_dir = '/data15/tcrawfor/run1_clusters_fullsurvey/'+ field + '/'
  tfer_function_dir = '/data/bleeml/optical_cluster_projects/run1_ymap_fullsurvey_simulation/1d_beam/'
  spt_map_dir = '/data15/tcrawfor/run1_clusters_ymap/'+ field
  mapfile =  '/data/bleeml/optical_cluster_projects/sptsz_field_info.sav'
  outdir = '/data/bleeml/for_amy/smg_checks/'


                                ;need calfacs!
  calfacs =[0.722787, 0.830593, 0.731330] ;Monica's correction to CR's numbers https://spt.uchicago.edu/trac/wiki/lps14_calibration 

  
  restore, mapfile, /v
  kp = where(field_info.name eq field)
  field_info = field_info[kp]
  radec0 = [field_info.RA0, field_info.DEC0]
  
  restore, spt_map_dir + '/coadds_latest.sav', /v
  reso_arcmin = 0.25
  ngrid_x = n_elements(maps[*,0,0])
  ngrid_y = n_elements(maps[0,*,0])
  ellgrid = make_fft_grid(reso_arcmin*!dtor/60.,ngrid_x,ngrid_y,/waven) ;making high-resolution (output resolution) ell grid

      
  restore, tfer_function_dir + field + '_xfer_function_with_beam.sav', /v
  psd_name = spt_map_dir + '/psd_'+field + '_'+band + '_wtd.sav'
  jk = mrdfits(apmask_dir + 'masks_allbands_0p8.fits', 1)
  apmasks = jk.APOD_MASK
  pmtemp = jk.PIXEL_MASK
  q = where(apmasks lt 0.01)
  apmasks[q]  =0

  
  camb2cl_k11,cl_all,ell=ellcmb ;CMB
  clcmb = cl_all[*,0]
  undefine, cl_all           

  profs = fltarr(40000)+1.0

;---------------------------
                                ;point sources:
  psmask_arcmin = 4.
  minrad_ps = 8.
  ptsrcfield = field
  if strpos(field,'2year') ne -1 then ptsrcfield = (strsplit(field,'_2year',/reg,/ext))[0]
  ptsrc_file = '/home/tcrawfor/spt_analysis/config/ptsrc_config_'+ptsrcfield+'_surveyclusters_cfind.txt'
  ptsrc_struct_radec = read_ptsrc_config('none',file=ptsrc_file)

  if proj eq 0 then begin
     ang2pix_proj0,ptsrc_struct_radec.ra,ptsrc_struct_radec.dec,[ngrid_x, ngrid_y],radec0,reso_arcmin,iptemp,xp=xps,yp=yps
     print,"Proj0 psource"
  endif

  kper =where(xps gt 0 and yps gt 0 and xps lt ngrid_x and yps lt ngrid_y)
  if kper[0] ne -1 then begin
     xps = xps[kper]
     yps = yps[kper]
  endif

  radpix = ptsrc_struct_radec.rad*3600./15.
  ptsrc_struct = {xpeak:xps,ypeak:yps,radpix:radpix}

                                ;dusty sources:
  clps_dusty = fltarr(n_elements(band))
  clps_dusty[0] = 4.9d-19       ; 
  clps_dusty[1] = 5.4e-18       ; S10
  clps_dusty[2] = 4.9e-17       ; cl = (10^-12)*dl/((3000*(3000+1)/(2*!pi)))  dl = 70 uk^2   print, 10^(-12.0)*(70)*2*!pi/3000/3001.
  
  
  for i=0, 2, 1 do begin  ;loop over the frequencies

     maps2use = maps[*,*,i]*calfacs[i]
     restore, psd_name[i]
     psds2use = psd*calfacs[i]
     bandssz=0
     dlsz2use=0
     clps2use = clps_dusty[i]

     ;transfer function at low kx
     trans_func = transfer_with_Beam[*,*,i]
     qq = where(ellgrid[*,0] lt 400)
     trans_func[qq,*] = 0
     psds2use[qq,*] = 1e9
     
     noise_bandsize_arcmin = 90.
     
     CLUSTERFIND_multiBAND_MULTIPROFILE, $
        maps2use, bandssz, profs, outtemp_2band, $
        noisepsd=psds2use, $
        trans_func_grids=trans_func, $
        clcmb=clcmb, $
        dlsz=dlsz2use, $
        clps=clps2use, $
        reso_arcmin=reso_arcmin, $
        szmaps_filt = szmaps_filt, $
        nsigma=40., $
        apod_mask=apmasks, $
        pixel_mask=pmtemp, $
        xpeak_ps=xpeak_ps, ypeak_ps=ypeak_ps, $
        ptsrc_struct=ptsrc_struct, $
        sigvals_ps=sigvals_ps, $
        maskrad_ps=psmask_arcmin, $
        interp_over_ps=interp_over_ps, $
        filt_area=filt_area,savemem=savemem,fracelscan=fracelscan,elscan=elscan, $
        ext_src_pix_file=ext_src_pix_file,external_profile_norm=external_profile_norm, $
        beam_profile=beam_profile, flatwt=1, $
        filtered_map=filtered_map, $
        OPTFILT2D=OPTFILT2D, $
        COMPUTEAREA=COMPUTEAREA, $
        stopit=stopit


                                ;well this code work, but it depends
                                ;on whether you use "solid angle" or
                                ;"computeara"; I think computearea is
                                ;correct.
     
     if 0 then begin 
                                ;and let's compute the uK- mJy factor
        ;DON'T USE THIS PART!!!
        reso = (reso_arcmin/60.0)*!pi/180.
        TCMB = 2.725
        h = 6.62607e-34
        kb = 1.38065e-23
        dk = 1./ngrid_x/reso
        c = 2.9979e8
        tauprime = OPTFILT2D*trans_func
        solid_angle =1./total(optfilt2d*tauprime)  / (dk^2)
        x = float(band[i])*1.0e9*h/(kb*Tcmb)
        conversion = solid_angle*1.0e26*(2*kb/c^2.0)*((kb*Tcmb/h)^2)*(x^4)*exp(x)/(exp(x)-1)^2.0
        conversion_factor = conversion*1.0e3/1.0e6 ;uK and mJy 
        uK_to_mjy = conversion_factor
     endif
     
     outfile = outdir + field + '_'+band[i] + '_check.sav'
     save, szmaps_filt, filtered_map, outtemp_2band, radec0, reso_arcmin,solid_angle, OPTFILT2D, computearea, solid_angle, file=outfile
  endfor

  ;/home/bleeml/spt_analysis/util/mjy_to_uk.pro

  
  stop
  stop
  return
end
  
