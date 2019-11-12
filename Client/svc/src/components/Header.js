import React, {Component} from 'react';
import {
    Heading, Text
} from "gestalt";
import 'gestalt/dist/gestalt.css';
import "bootstrap/dist/css/bootstrap.css";

const Header = () =>{
    return(
        <Text align="center" bold>
            <Heading size="md" color="white">
                Swift Video Coloring
            </Heading>
        </Text>
    )
}

export default Header;