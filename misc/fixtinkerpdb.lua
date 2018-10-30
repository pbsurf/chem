-- TINKER xyzpdb applied to a multiple structure xyz file generated
--  by TINKER archive is equiv to concatenating a bunch of pdb files,
--  which is invalid.  This script creates a valid pdb file from the mess.
f1 = io.open(arg[1]);
f2 = io.open(arg[2], "w");

mdlcnt = 0;
for line in f1:lines() do
  str3 = string.sub(line, 1, 3);
  str6 = string.sub(line, 1, 6);
  if mdlcnt > 0 then
    if str3 == "END" then
      f2:write("ENDMDL\n");
    elseif str6 == "HEADER" then
      mdlcnt = mdlcnt + 1;
      f2:write(string.format("MODEL        %d\n", mdlcnt));
      io.write('.');
    elseif str6 ~= "COMPND" and str6 ~= "SOURCE" then
      f2:write(line.."\n");
    end
  else
    if str3 == "ATO" or str6 == "HETATM" then
      nummdlpos = f2:seek();
      f2:write("NUMMDL    ?            \nMODEL        1\n");
      mdlcnt = 1;
      io.write('.');
    end
    f2:write(line.."\n");
  end
end

f2:write("END\n");
-- seek to NUMMDL line
if nummdlpos then
  f2:seek("set", nummdlpos);
  f2:write(string.format("NUMMDL    %d", mdlcnt));
else
  print('No "ATOM" keywords found; not a PDB file?');
end
f1:close();
f2:close();
