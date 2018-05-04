function scattering_coefficient_method_parameters =...
    instantiate_scattering_coefficient_method(scattering_coefficient_method,...
    maximum_value, minimum_value, random_generator,...
    configure_random_generator, beta)
%INSTANTIATE_SCATTERING_COEFFICIENT_METHOD  Assign values to parameters of the
%specified method for generation of scattering coefficients.
%   Inputs:
%   -|scattering_coefficient_method|: handle of the function which is specified
%   as the scattering coefficient method.
%   -|maximum_value|: upper end of interval for scattering coefficient values.
%   -|minimum_value|: lower end of interval for scattering coefficient values.
%   -|random_genarator|: type of random number generator.
%   -|configure_random_generator|: flag indicating whether to configure random
%   number generator or not.
%   -|beta|: fixed value of scattering coefficient.
%
%   Outputs:
%   -|scattering_coefficient_method_parameters|: structure containing the
%   various parameters of the scattering coefficient method as its fields.

switch func2str(scattering_coefficient_method)
    case 'scattering_coefficient_random'
        assert(maximum_value >= minimum_value);
        assert(minimum_value >= 0);
        scattering_coefficient_method_parameters.maximum_value = maximum_value;
        scattering_coefficient_method_parameters.minimum_value = minimum_value;
        scattering_coefficient_method_parameters.random_generator =...
            random_generator;
        scattering_coefficient_method_parameters.configure_random_generator =...
            configure_random_generator;
        
    case 'scattering_coefficient_fixed'
        assert(beta >= 0);
        scattering_coefficient_method_parameters.beta = beta;
end

end

