document.addEventListener('DOMContentLoaded', function() {
    const script = document.createElement('script');
    script.src = `https://openapi.map.naver.com/openapi/v3/maps.js?ncpClientId=${config.apikey}`;
    script.async = true;
    script.onload = function() {
        const { naver } = window;
        if (!naver) return;

        var map = new naver.maps.Map('map', {
            center: new naver.maps.LatLng(37.54523, 126.9643),
            zoom: 17
        });
        // var marker = new naver.maps.Marker({
        //     position: new naver.maps.LatLng(37.54523, 126.9643),
        //     map: map
        // });
        var locations = [
            { lat: 37.545249, lng: 126.964874 },
            { lat: 37.54449, lng: 126.9686 },
            { lat: 37.5445509, lng: 126.9669017 },
            { lat: 37.5452300, lng: 126.9670614 },
            // 추가적인 위치를 여기에 추가
        ];

        // 각 위치에 마커 추가
        locations.forEach(function(location) {
            new naver.maps.Marker({
                position: new naver.maps.LatLng(location.lat, location.lng),
                map: map
            });
        });

    };
    document.head.appendChild(script);
  });
  