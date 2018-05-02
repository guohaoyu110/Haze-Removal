function atmospheric_light_method_parameters =...
    instantiate_atmospheric_light_method(atmospheric_light_method,...
    maximum_intensity, minimum_intensity, random_generator,...
    configure_random_generator, c)
%INSTANTIATE_ATMOSPHERIC_LIGHT_METHOD  Assign values to parameters of the
%specified method for generation of atmospheric light values.
%   Inputs:
%   -|atmospheric_light_method|: handle of the function which is specified as
%   the atmospheric light method.
%   -|maximum_intensity|: upper end of interval for atmospheric light values.
%   -|minimum_intensity|: lower end of interval for atmospheric light values.
%   -|random_genarator|: type of random number generator.
%   -|configure_random_generator|: flag indicating whether to configure random
%   number generator or not.
%   -|c|: fixed value of atmospheric light intensity.
%
%   Outputs:
%   -|atmospheric_light_method_parameters|: structure containing the various
%   parameters of the atmospheric light method as its fields.

switch func2str(atmospheric_light_method)
    case 'atmospheric_light_random'
        assert(maximum_intensity >= minimum_intensity);
        assert(maximum_intensity <= 1);
        assert(minimum_intensity >= 0);
        atmospheric_light_method_parameters.maximum_intensity =...
            maximum_intensity;
        atmospheric_light_method_parameters.minimum_intensity =...
            minimum_intensity;
        atmospheric_light_method_parameters.random_generator = random_generator;
        atmospheric_light_method_parameters.configure_random_generator =...
            configure_random_generator;
        
    case 'atmospheric_light_fixed'
        assert(c >= 0 && c <= 1);
        atmospheric_light_method_parameters.c = c;
end

end

