package oracleai.services;

import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.Base64;
import java.util.Collections;

@Service
public class ORDSCalls {

    public static String callAnalyzeImageInline(String ordsEndpoint, String visionServiceIndpoint,
                                             String compartmentOcid, MultipartFile imageFile)
            throws Exception {
        RestTemplate restTemplate = new RestTemplate();
            String base64ImageData =  Base64.getEncoder().encodeToString(imageFile.getBytes());
        String jsonBody = String.format("{\"p_compartment_ocid\": \"%s\", \"p_endpoint\": \"%s\", \"p_image_data\": \"%s\"}",
                compartmentOcid, visionServiceIndpoint, base64ImageData);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<String> requestEntity = new HttpEntity<>(jsonBody, headers);
        ResponseEntity<String> response = restTemplate.exchange(ordsEndpoint, HttpMethod.POST, requestEntity, String.class);
        return response.getBody();
    }


    //As written only supports on feature type per call
    public static String analyzeImageInObjectStore(
            String ordsEndpoint, String visionServiceEndpoint, String compartmentOcid,
            String bucketName, String namespaceName, String objectName, String featureType, String label) {
        System.out.println("ORDSCalls.analyzeImageInObjectStore");
        System.out.println("ordsEndpoint = " + ordsEndpoint + ", visionServiceEndpoint = " + visionServiceEndpoint + ", compartmentOcid = " + compartmentOcid + ", bucketName = " + bucketName + ", namespaceName = " + namespaceName + ", objectName = " + objectName + ", featureType = " + featureType + ", label = " + label);
        RestTemplate restTemplate = new RestTemplate();
        String jsonPayload = String.format(
                "{\"p_bucketname\": \"%s\", \"p_compartment_ocid\": \"%s\", \"p_endpoint\": \"%s\", " +
                        "\"p_namespacename\": \"%s\", \"p_objectname\": \"%s\", \"p_featuretype\": \"%s\", \"p_label\": \"%s\"}",
                bucketName, compartmentOcid, visionServiceEndpoint, namespaceName, objectName, featureType, label);
//        jsonPayload = "{\"p_bucketname\": \"doc\",\n" +
//                "\"p_compartment_ocid\": \"ocid1.compartment.oc1..aaaaaaaafnah3ogykjsg34qruhixhb2drls6zhsejzm7mubi2i5qj66slcoq\",\n" +
//                "\"p_endpoint\": \"https://vision.aiservice.us-ashburn-1.oci.oraclecloud.com\",\n" +
//                "\"p_namespacename\": \"oradbclouducm\",\n" +
//                "\"p_objectname\": \"bloodsugarreport.jpeg\",\n" +
//                "\"p_featuretype\": \"TEXT_DETECTION\",\n" +
//                "\"p_label\": \"MedicalReportSummary\"}";
        System.out.println("ORDSCalls.analyzeImageInObjectStore jsonPayload:" + jsonPayload);
        HttpHeaders headers = new HttpHeaders();
        headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<String> entity = new HttpEntity<>(jsonPayload, headers);
        ResponseEntity<String> response = restTemplate.exchange(ordsEndpoint, HttpMethod.POST, entity, String.class);
        System.out.println("ORDSCalls.analyzeImageInObjectStore response.getBody():" + response.getBody());
        return response.getBody();
    }

    public static String executeDynamicSQL(
            String ordsEndpoint, String sql) {
        System.out.println("executeDynamicSQL ordsEndpoint = " + ordsEndpoint + ", sql = " + sql);
        RestTemplate restTemplate = new RestTemplate();
        String jsonPayload = String.format( "{\"p_sql\": \"%s\"}",  sql);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<String> entity = new HttpEntity<>(jsonPayload, headers);
        ResponseEntity<String> response = restTemplate.exchange(ordsEndpoint, HttpMethod.POST, entity, String.class);
        System.out.println("ORDSCalls.analyzeImageInObjectStore response.getBody():" + response.getBody());
        return response.getBody();
    }


}

